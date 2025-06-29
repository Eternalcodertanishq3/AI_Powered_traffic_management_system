import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageSequence
import os
import csv
from datetime import datetime
import sys
import threading
import time

# --- Configuration Flag for Advanced UI Features ---
ENABLE_ADVANCED_UI_FEATURES = True

# Add the srcfolder to the Python path to allow importing modules correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir) # <<<--- TYPO FIXED HERE!

# --- Import SCENE_CLASSES from the central constants.py ---
try:
    from .constants import SCENE_CLASSES
    print("Successfully imported SCENE_CLASSES from constants.py.")
except ImportError as e:
    messagebox.showerror("Import Error",
                          f"Could not import SCENE_CLASSES from constants.py: {e}\n"
                          "Please ensure constants.py exists in srcfolder and defines SCENE_CLASSES.")
    SCENE_CLASSES = ["normal_traffic", "accident", "dense_traffic", "sparse_traffic"] # Fallback if constants.py is missing/incorrect


# --- IMPORTANT: Try to import the model's prediction logic from detection_model.py ---
try:
    from .detection_model import get_scene_prediction
    print("Successfully imported get_scene_prediction from detection_model.py.")
    MODEL_PREDICTION_AVAILABLE = True and ENABLE_ADVANCED_UI_FEATURES
except ImportError as e:
    messagebox.showerror("Import Error",
                          f"Could not import get_scene_prediction from detection_model.py: {e}\n"
                          "Automatic scene predictions from model will be disabled. Ensure detection_model.py is correct and scene_classifier.pth exists.")
    MODEL_PREDICTION_AVAILABLE = False
except Exception as e:
    messagebox.showerror("Model Loading Error",
                          f"An error occurred while setting up model prediction: {e}\n"
                          "Automatic scene predictions from model will be disabled. Check model dependencies.")
    MODEL_PREDICTION_AVAILABLE = False


class ImageLabelingTool:
    def __init__(self, master):
        self.master = master
        master.title("Traffic Scene Annotation Tool (Automated Filename Labeling)")
        master.geometry("1400x900")
        master.resizable(True, True)
        master.configure(bg="#F0F2F5")

        self.image_paths = []
        self.current_image_idx = -1
        self.current_image_tk = None
        self.original_image = None
        
        self.output_csv_path = "annotations.csv"
        self.output_csv_path_var = tk.StringVar(master, value=self.output_csv_path)
        self.output_csv_path_var.trace_add("write", self._update_output_csv_path)

        self.annotations = {} # Stores {image_path: label} for the CURRENTLY LOADED CSV

        self.animation_frames = []
        self.animation_index = 0
        self.animation_id = None
        if ENABLE_ADVANCED_UI_FEATURES:
            gif_path = os.path.join(current_dir, "assets", "scan_animation.gif")
            self.load_animation_gif(gif_path)

        self.setup_ui()
        self.load_annotations_from_csv() 

    def _update_output_csv_path(self, *args):
        """Callback function to update the internal output_csv_path when StringVar changes."""
        new_path = self.output_csv_path_var.get()
        if new_path and new_path != self.output_csv_path:
            self.output_csv_path = new_path
            print(f"[INFO] Output CSV path changed to: {self.output_csv_path}. Reloading annotations.")
            self.load_annotations_from_csv()

    def load_animation_gif(self, gif_path):
        if not ENABLE_ADVANCED_UI_FEATURES: return
        try:
            if os.path.exists(gif_path):
                gif_image = Image.open(gif_path)
                for frame in ImageSequence.Iterator(gif_image):
                    self.animation_frames.append(ImageTk.PhotoImage(frame.copy().resize((50,50), Image.LANCZOS)))
                print(f"[INFO] Loaded animation GIF from {gif_path}")
            else:
                print(f"[WARNING] Animation GIF not found at {gif_path}. Using text indicator.")
                self.animation_frames = []
        except Exception as e:
            print(f"[ERROR] Error loading animation GIF: {e}. Using text indicator.")
            self.animation_frames = []

    def setup_ui(self):
        top_frame = tk.Frame(self.master, padx=10, pady=10, bg="#E0E5EC", bd=2, relief=tk.FLAT)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.load_button = ttk.Button(top_frame, text="Load Image Folder", command=self.load_folder, style='TButton')
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)

        tk.Label(top_frame, text="Output CSV File:", bg="#E0E5EC", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(20,0), pady=5)
        self.output_csv_entry = ttk.Entry(top_frame, textvariable=self.output_csv_path_var, width=30)
        self.output_csv_entry.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_csv_button = ttk.Button(top_frame, text="Save Annotations CSV", command=self.save_annotations_to_csv, style='TButton')
        self.save_csv_button.pack(side=tk.RIGHT, padx=10, pady=5)

        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10), padding=8, background='#4CAF50', foreground='black')
        style.map('TButton', background=[('active', '#66BB6A')])


        main_frame = tk.Frame(self.master, padx=15, pady=15, bg="#F0F2F5")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=4)
        main_frame.grid_columnconfigure(1, weight=1)

        self.canvas = tk.Canvas(main_frame, bg="#2C3E50", bd=2, relief=tk.SUNKEN, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas.bind("<Configure>", self.resize_image)

        control_frame = tk.Frame(main_frame, padx=20, pady=20, bg="#FFFFFF", bd=2, relief=tk.RAISED)
        control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        control_frame.grid_rowconfigure(len(SCENE_CLASSES) + 10, weight=1)
        control_frame.grid_columnconfigure(0, weight=1)

        tk.Label(control_frame, text="Predicted Scene:", font=("Arial", 13, "bold"), bg="#FFFFFF", fg="#34495E").pack(pady=(10, 5))
        self.predicted_label = tk.Label(control_frame, text="N/A", font=("Arial", 16, "italic"), fg="#2980B9", bg="#FFFFFF")
        self.predicted_label.pack(pady=(0, 5))
        
        self.prediction_status_label = tk.Label(control_frame, text="", font=("Arial", 10), fg="gray", bg="#FFFFFF")
        self.prediction_status_label.pack()

        self.animation_canvas = tk.Canvas(control_frame, width=50, height=50, bg="#FFFFFF", highlightthickness=0)
        self.animation_canvas.pack(pady=5)
        if not ENABLE_ADVANCED_UI_FEATURES:
            self.animation_canvas.pack_forget()


        tk.Label(control_frame, text="Select True Scene:", font=("Arial", 13, "bold"), bg="#FFFFFF", fg="#34495E").pack(pady=(20, 5))
        
        self.selected_label_var = tk.StringVar(self.master)
        self.selected_label_var.set("Select a Label")

        self.label_dropdown = ttk.OptionMenu(control_frame, self.selected_label_var, "Select a Label", *SCENE_CLASSES)
        self.label_dropdown.pack(pady=(0, 15), fill=tk.X, ipadx=5, ipady=5)

        self.confirm_button = ttk.Button(control_frame, text="Confirm & Next (Enter)", command=self.confirm_annotation, style='TButton')
        self.confirm_button.pack(pady=10, fill=tk.X, ipadx=10, ipady=10)
        self.master.bind("<Return>", lambda event: self.confirm_annotation())

        nav_frame = tk.Frame(control_frame, bg="#FFFFFF")
        nav_frame.pack(pady=(20,10), fill=tk.X)
        self.prev_button = ttk.Button(nav_frame, text="Previous (Left Arrow)", command=self.load_previous_image, style='TButton')
        self.prev_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.next_button = ttk.Button(nav_frame, text="Next (Right Arrow)", command=self.load_next_image, style='TButton')
        self.next_button.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=2)
        self.master.bind("<Left>", lambda event: self.load_previous_image())
        self.master.bind("<Right>", lambda event: self.load_next_image())


        bottom_frame = tk.Frame(self.master, padx=10, pady=5, bg="#E0E5EC", bd=2, relief=tk.SUNKEN)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(bottom_frame, text="Ready.", bg="#E0E5EC", anchor="w", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.current_file_label = tk.Label(bottom_frame, text="", bg="#E0E5EC", anchor="e", font=("Arial", 10, "italic"))
        self.current_file_label.pack(side=tk.RIGHT, fill=tk.X, padx=5)

        self.update_status_bar()
    
    def load_annotations_from_csv(self):
        """Loads annotations from the currently set self.output_csv_path."""
        self.annotations = {} # Clear current annotations before loading new ones
        if os.path.exists(self.output_csv_path):
            try:
                with open(self.output_csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header and 'image_path' in header and 'scene_label' in header:
                        try:
                            path_idx = header.index('image_path')
                            label_idx = header.index('scene_label')
                        except ValueError:
                            messagebox.showwarning("CSV Format Error", "CSV header missing 'image_path' or 'scene_label'.")
                            return

                        for row in reader:
                            if len(row) > max(path_idx, label_idx):
                                # --- FIX: Normalize path when loading from CSV ---
                                image_path_raw = row[path_idx]
                                image_path_normalized = image_path_raw.replace('\\', '/') # Replace backslashes with forward slashes
                                self.annotations[image_path_normalized] = row[label_idx]
                    else:
                         messagebox.showwarning("CSV Format Error", "CSV is empty or header is incorrect.")

                print(f"[INFO] Loaded {len(self.annotations)} existing annotations from {self.output_csv_path}")
                self.update_status_bar(f"Loaded {len(self.annotations)} existing annotations from {os.path.basename(self.output_csv_path)}.")
            except Exception as e:
                messagebox.showerror("Error Loading CSV", f"Could not load existing annotations from {self.output_csv_path}: {e}")
        else:
            self.update_status_bar(f"No existing annotations CSV '{os.path.basename(self.output_csv_path)}' found. Starting fresh.")

    def _extract_label_from_filename(self, filename):
        """
        Attempts to extract a scene label from the image filename.
        Returns the matching SCENE_CLASS or None if no match.
        This version is highly tuned for the user's provided filename patterns.
        """
        base_name = os.path.splitext(filename)[0].lower()
        
        label_prefixes = {
            "accident1": "accident",
            "accident": "accident",
            "nonaccident": "normal_traffic",
        }

        for prefix, scene_class_mapping in label_prefixes.items():
            if base_name.startswith(prefix):
                if scene_class_mapping in SCENE_CLASSES:
                    print(f"[DEBUG] Filename '{filename}' matched prefix '{prefix}', mapping to '{scene_class_mapping}'")
                    return scene_class_mapping
                else:
                    print(f"[WARNING] Filename '{filename}' matched prefix '{prefix}' but mapped scene class '{scene_class_mapping}' is NOT in SCENE_CLASSES.")
                    return None

        print(f"[DEBUG] No scene label prefix found in filename '{filename}'")
        return None

    def load_folder(self, folder_path=None): # Added optional folder_path argument
        if folder_path is None: # If not provided, open dialog
            folder_path = filedialog.askdirectory(title="Select Image Folder")
        
        if folder_path:
            self.image_paths = []
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        # --- FIX: Normalize path when collecting image paths ---
                        normalized_path = os.path.join(root, file).replace('\\', '/')
                        self.image_paths.append(normalized_path)
            self.image_paths.sort()

            if not self.image_paths:
                messagebox.showinfo("No Images", f"No supported images ({', '.join(valid_extensions)}) found in the selected folder.")
                self.current_image_idx = -1
                self.clear_display()
            else:
                base_folder_name = os.path.basename(folder_path)
                suggested_csv_name = None

                if "video_frames" in folder_path.lower() or "_frames" in base_folder_name.lower():
                    if "_frames" in base_folder_name.lower():
                        clean_name = base_folder_name.replace("_frames", "").replace("_", "-")
                        suggested_csv_name = f"{clean_name}_video_annotations.csv"
                    else:
                        suggested_csv_name = f"{base_folder_name}_video_annotations.csv"
                else:
                    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
                    relative_path = os.path.relpath(folder_path, project_root)
                    suggested_csv_name = f"{relative_path.replace(os.sep, '_')}_annotations.csv"
                    suggested_csv_name = suggested_csv_name.replace("data_", "").replace("train_", "").replace("image_2.1_", "").replace("traf_acci_data_", "")
                    
                    if not suggested_csv_name or suggested_csv_name.startswith('_'):
                        suggested_csv_name = f"{base_folder_name}_annotations.csv"

                if suggested_csv_name:
                    self.output_csv_path_var.set(suggested_csv_name.lower())
                else:
                    self.output_csv_path_var.set("annotations.csv")
                
                self.apply_smart_bulk_labeling()

                self.current_image_idx = 0
                self.load_image()
                self.update_status_bar(f"Loaded {len(self.image_paths)} images from {folder_path}")


    def apply_smart_bulk_labeling(self):
        """
        Iterates through loaded images and attempts to auto-label them based on filename.
        Only labels images not already present in self.annotations.
        Adds successfully labeled images to self.annotations and saves to CSV.
        """
        bulk_labeled_count = 0
        print(f"[INFO] Starting smart bulk labeling for {len(self.image_paths)} images.")
        for image_path in self.image_paths:
            if image_path not in self.annotations: 
                filename = os.path.basename(image_path)
                extracted_label = self._extract_label_from_filename(filename)
                
                if extracted_label:
                    self.annotations[image_path] = extracted_label
                    bulk_labeled_count += 1
                    print(f"[INFO] Auto-labeled: '{filename}' as '{extracted_label}'")
        
        if bulk_labeled_count > 0:
            self.save_annotations_to_csv()
            messagebox.showinfo("Smart Bulk Labeling Complete",
                                f"Automatically labeled {bulk_labeled_count} images based on filenames and saved to {os.path.basename(self.output_csv_path)}. "
                                "You can now hit ENTER repeatedly to confirm these labels quickly. Review any images that still require manual input!")
        else:
            print("[INFO] No new images were automatically labeled from filenames in this batch (or all were already labeled).")


    def load_image(self):
        if not self.image_paths:
            self.clear_display()
            return

        if self.animation_id:
            self.master.after_cancel(self.animation_id)
            self.animation_id = None
        self.animation_canvas.delete("all")

        if self.current_image_idx < 0:
            self.current_image_idx = 0
        elif self.current_image_idx >= len(self.image_paths):
            self.current_image_idx = len(self.image_paths) - 1

        image_path = self.image_paths[self.current_image_idx]
        self.current_file_label.config(text=f"File: {os.path.basename(image_path)}")

        if image_path in self.annotations:
            self.selected_label_var.set(self.annotations[image_path])
            self.predicted_label.config(text="Pre-Labeled (via Filename)", fg="#800080")
            self.prediction_status_label.config(text="Confirmed by Filename / Previous Annotation. Press Enter to proceed.", fg="#800080")
            self.display_image_on_canvas(Image.open(image_path).convert("RGB"))
            self.update_status_bar()
            if self.animation_id:
                self.master.after_cancel(self.animation_id)
                self.animation_id = None
                self.animation_canvas.delete("all")
            return

        try:
            pil_image = Image.open(image_path).convert("RGB")
            self.original_image = pil_image

            if MODEL_PREDICTION_AVAILABLE:
                self.predicted_label.config(text="Analyzing...", fg="gray")
                self.prediction_status_label.config(text="Running model prediction...", fg="gray")
                self.master.update_idletasks()

                if self.animation_frames:
                    self.animate_scan()
                else:
                    self.animate_text_scan(0)
                
                threading.Thread(target=self._run_prediction_and_update_ui, args=(pil_image, image_path)).start()
            else:
                self.predicted_label.config(text="N/A", fg="gray")
                self.prediction_status_label.config(text="Manual Mode. No model prediction.", fg="blue")
                self.selected_label_var.set("Select a Label") 
                self.display_image_on_canvas(self.original_image)
                self.update_status_bar()


        except Exception as e:
            messagebox.showerror("Error Loading Image", f"Could not load image {image_path}: {e}")
            self.clear_display()
            if self.animation_id: self.master.after_cancel(self.animation_id)

    def animate_scan(self):
        if not ENABLE_ADVANCED_UI_FEATURES: return
        if not self.animation_frames: return

        frame = self.animation_frames[self.animation_index]
        self.animation_canvas.delete("all")
        self.animation_item = self.animation_canvas.create_image(
            self.animation_canvas.winfo_width() / 2,
            self.animation_canvas.winfo_height() / 2,
            image=frame, anchor=tk.CENTER
        )
        self.animation_index = (self.animation_index + 1) % len(self.animation_frames)
        self.animation_id = self.master.after(100, self.animate_scan)

    def animate_text_scan(self, dot_count):
        if not ENABLE_ADVANCED_UI_FEATURES: return
        if self.animation_id:
            dots = "." * (dot_count % 4)
            self.prediction_status_label.config(text=f"Analyzing{dots}")
            self.animation_id = self.master.after(300, self.animate_text_scan, dot_count + 1)


    def _run_prediction_and_update_ui(self, pil_image, image_path):
        if not ENABLE_ADVANCED_UI_FEATURES: return
        predicted_label = "N/A"
        predicted_confidence = 0.0

        if MODEL_PREDICTION_AVAILABLE:
            time.sleep(0.1)
            try:
                predicted_label, predicted_confidence = get_scene_prediction(pil_image)
            except Exception as e:
                print(f"[ERROR] Model prediction failed: {e}")
                predicted_label = "Model Error"
                predicted_confidence = 0.0
        else:
            predicted_label = "N/A (Model Disabled)"
            predicted_confidence = 0.0
            time.sleep(0.1)

        self.master.after(0, self._update_prediction_ui, predicted_label, predicted_confidence, image_path)
        self.master.after(0, self.display_image_on_canvas, self.original_image)
        self.master.after(0, self.update_status_bar)

        if self.animation_id:
            self.master.after_cancel(self.animation_id)
            self.animation_id = None
            self.animation_canvas.delete("all")


    def _update_prediction_ui(self, predicted_label, predicted_confidence, image_path):
        if not ENABLE_ADVANCED_UI_FEATURES: return

        AUTO_ACCEPT_THRESHOLD = 0.95
        SUGGEST_REVIEW_THRESHOLD = 0.75
        MANUAL_LABEL_THRESHOLD = 0.40

        feedback_color = "#34495E"
        feedback_text = ""

        if image_path in self.annotations: 
            self.selected_label_var.set(self.annotations[image_path])
            feedback_color = "#800080"
            feedback_text = "Previously Annotated (Manual Override)"
            self.predicted_label.config(text=f"Model: {predicted_label} ({predicted_confidence:.1%})", fg=feedback_color)
        elif predicted_confidence >= AUTO_ACCEPT_THRESHOLD:
            self.selected_label_var.set(predicted_label)
            feedback_color = "#28A745"
            feedback_text = "High Confidence - Auto-Select (Press Enter to Confirm)"
            self.predicted_label.config(text=f"Model: {predicted_label} ({predicted_confidence:.1%})", fg=feedback_color)
        elif predicted_confidence >= SUGGEST_REVIEW_THRESHOLD:
            self.selected_label_var.set(predicted_label)
            feedback_color = "#17A2B8"
            feedback_text = "Good Confidence - Review Recommended (Confirm/Correct)"
            self.predicted_label.config(text=f"Model: {predicted_label} ({predicted_confidence:.1%})", fg=feedback_color)
        elif predicted_confidence > MANUAL_LABEL_THRESHOLD:
            self.selected_label_var.set(predicted_label)
            feedback_color = "#FFC107"
            feedback_text = "Moderate Confidence - Manual Labeling Advised!"
            self.predicted_label.config(text=f"Model: {predicted_label} ({predicted_confidence:.1%})", fg=feedback_color)
        else:
            self.selected_label_var.set("Select a Label")
            feedback_color = "#DC3545"
            feedback_text = "Low Confidence - MANUAL LABELING REQUIRED!"
            self.predicted_label.config(text=f"Model: {predicted_label} ({predicted_confidence:.1%})", fg=feedback_color)
            messagebox.showwarning("Manual Labeling Needed",
                                   f"Model confidence is very low ({predicted_confidence:.1%}) for '{predicted_label}'. "
                                   "Please manually select the correct label. This is crucial for model improvement!")

        self.prediction_status_label.config(text=f"Status: {feedback_text}", fg=feedback_color)


    def display_image_on_canvas(self, pil_image):
        if self.canvas.winfo_width() == 0 or self.canvas.winfo_height() == 0:
            self.master.after(100, lambda: self.display_image_on_canvas(pil_image))
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        img_width, img_height = pil_image.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        self.current_image_tk = ImageTk.PhotoImage(resized_image)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_width / 2, canvas_height / 2,
                                 image=self.current_image_tk, anchor=tk.CENTER)

    def resize_image(self, event=None):
        if self.original_image:
            self.display_image_on_canvas(self.original_image)

    def load_next_image(self):
        if self.current_image_idx != -1 and self.image_paths:
            self.confirm_annotation_silent()

        if self.current_image_idx < len(self.image_paths) - 1:
            self.current_image_idx += 1
            self.load_image()
        else:
            self.update_status_bar("Reached end of images. All images processed.")
            messagebox.showinfo("End of Images", "You have reached the end of the image list.")
            self.save_annotations_to_csv()


    def load_previous_image(self):
        if self.current_image_idx != -1 and self.image_paths:
            self.confirm_annotation_silent()

        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_image()
        else:
            self.update_status_bar("Reached beginning of images.")
            messagebox.showinfo("Beginning of Images", "You are at the first image.")

    def confirm_annotation(self):
        """Confirms the current annotation and moves to the next image."""
        if self.current_image_idx == -1 or not self.image_paths:
            messagebox.showwarning("No Image", "Please load an image folder first to annotate.")
            return

        selected_label = self.selected_label_var.get()
        if selected_label == "Select a Label" or selected_label not in SCENE_CLASSES:
            messagebox.showwarning("Missing Label", "Please select a valid scene label before confirming.")
            return

        image_path = self.image_paths[self.current_image_idx]
        self.annotations[image_path] = selected_label
        
        self.save_annotations_to_csv()
        self.update_status_bar(f"Annotated: '{os.path.basename(image_path)}' as '{selected_label}' and saved to {os.path.basename(self.output_csv_path)}.")
        self.load_next_image()

    def confirm_annotation_silent(self):
        """Confirms the current annotation without user prompt or moving to next image."""
        if self.current_image_idx == -1 or not self.image_paths:
            return 

        selected_label = self.selected_label_var.get()
        if selected_label != "Select a Label" and selected_label in SCENE_CLASSES:
            image_path = self.image_paths[self.current_image_idx]
            if image_path not in self.annotations or self.annotations[image_path] != selected_label:
                self.annotations[image_path] = selected_label
                self.save_annotations_to_csv()
                print(f"[INFO] Saved (silent): '{os.path.basename(image_path)}' as '{selected_label}'")


    def save_annotations_to_csv(self):
        """Saves annotations to the currently set self.output_csv_path."""
        try:
            os.makedirs(os.path.dirname(self.output_csv_path) or '.', exist_ok=True)
            with open(self.output_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'scene_label', 'timestamp'])
                for path, label in self.annotations.items():
                    # Normalize path before writing to CSV
                    normalized_path = path.replace('\\', '/')
                    writer.writerow([normalized_path, label, datetime.now().isoformat()])
            self.update_status_bar(f"Annotations saved to {self.output_csv_path} ({len(self.annotations)} entries).")
            print(f"[INFO] Annotations successfully saved to {self.output_csv_path}")
        except Exception as e:
            messagebox.showerror("Error Saving CSV", f"Could not save annotations to {self.output_csv_path}: {e}")
            print(f"[ERROR] Failed to save annotations: {e}")


    def clear_display(self):
        self.canvas.delete("all")
        self.predicted_label.config(text="N/A", fg="gray")
        self.prediction_status_label.config(text="", fg="gray")
        if ENABLE_ADVANCED_UI_FEATURES:
            self.animation_canvas.delete("all")
            if self.animation_id: self.master.after_cancel(self.animation_id)
            self.animation_id = None
        self.selected_label_var.set("Select a Label")
        self.status_label.config(text="No images loaded.")
        self.current_file_label.config(text="")
        self.current_image_tk = None
        self.original_image = None

    def update_status_bar(self, message=None):
        if message:
            self.status_label.config(text=message)
        else:
            total = len(self.image_paths)
            current = self.current_image_idx + 1 if total > 0 else 0
            annotated_count = len(self.annotations)
            self.status_label.config(text=f"Image {current}/{total} | Annotated: {annotated_count} | CSV: {os.path.basename(self.output_csv_path)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelingTool(root)
    root.mainloop()

