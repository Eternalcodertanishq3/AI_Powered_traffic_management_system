import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import time
from collections import deque
from typing import Dict, Any, List, Tuple
from datetime import datetime
import math
import random 

# Add the srcfolder to the Python path to allow importing modules correctly
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import constants and the new scene prediction function from detection_model
from .constants import SCENE_CLASSES, YOLO_MODEL_PATH, YOLO_DETECTION_CONFIDENCE, YOLO_CLASSES, BBOX_COLORS
from .detection_model import get_scene_prediction

# --- Configuration ---
# VIDEO INPUT SOURCE CONFIGURATION ──────────────────────────────────
# UNCOMMENT only ONE of the following options:
# VIDEO_INPUT_SOURCE = "WEBCAM"
VIDEO_INPUT_SOURCE = r"C:/Personal Projects/AI_Powered_traffic_management_system/data/gettyimages-1936679257-640_adpp.mp4" # Example video path
# VIDEO_INPUT_SOURCE = r"C:/Personal Projects/AI_Powered_traffic_management_system/data/sample_traffic_video.mp4" # Another example
# ───────────────────────────────────────────────────────────────────

# Frame processing interval (process every Nth frame for performance)
FRAME_PROCESS_INTERVAL = 1

# --- SCENE CLASSIFICATION & ALERTING LOGIC CONFIG ---
SCENE_SMOOTHING_WINDOW_SIZE = 20 
MIN_SCENE_CONFIDENCE_DISPLAY = 0.50 

ACCIDENT_CONFIDENCE_THRESHOLD_OBSERVE = 0.70 
ACCIDENT_CONFIDENCE_THRESHOLD_WARN = 0.90 
ACCIDENT_CONFIDENCE_THRESHOLD_CRITICAL_ALERT = 0.95 
ACCIDENT_PERSISTENCE_FRAMES_WARN = 10 
ACCIDENT_PERSISTENCE_FRAMES_CRITICAL = 15 
ALERT_COOLDOWN_SECONDS = 30 

DENSE_TRAFFIC_CAR_COUNT_THRESHOLD_FOR_FALSE_POSITIVE = 15 
ACCIDENT_IMPACT_OBJECTS = ["stalled_vehicle", "debris_on_road", "overturned_vehicle", "fire", "truck", "bus"]
MIN_IMPACT_OBJECTS_FOR_ACCIDENT = 1 


# --- YOLO DETECTION CONFIG ---
YOLO_DETECTION_CONFIDENCE_OVERRIDE = 0.25 

# --- UI Styling (Iron Man / JARVIS Vibe) ---
# RGBA colors for transparency
HUD_BLUE_DARK = (10, 20, 50, 220)       
HUD_BLUE_MEDIUM = (20, 60, 100, 180)    
HUD_BLUE_LIGHT = (30, 144, 255, 180)    
HUD_CYAN_LIGHT = (0, 255, 255, 180)     
HUD_GREEN_LIGHT = (0, 255, 127, 180)    
HUD_YELLOW_ACCENT = (255, 255, 0, 180)  
HUD_RED_CRITICAL = (255, 69, 0, 180)    
HUD_TEXT_COLOR_PRIMARY = (220, 240, 255, 255) 
HUD_TEXT_COLOR_SECONDARY = (180, 200, 255, 255) 
HUD_TEXT_COLOR_HIGHLIGHT = (0, 255, 255, 255) 
HUD_OUTLINE_WIDTH_BASE = 2 # Base thickness, will scale
HUD_CORNER_RADIUS_BASE = 15 # Base radius, will scale

# Specific colors for scene labels (RGB tuples for PIL)
SCENE_LABEL_COLORS = {
    "normal_traffic": (0, 255, 100),       
    "dense_traffic": (255, 200, 0),      
    "sparse_traffic": (0, 150, 255),     
    "accident": (255, 50, 50),             
    "emergency_vehicle_passing": (255, 100, 0), 
    "road_block": (255, 80, 0),          
    "stalled_vehicle": (255, 150, 0),    
    "pedestrian_crossing": (100, 255, 255), 
    "fire": (255, 0, 0),                 
    "adverse_weather_rain": (100, 180, 255), 
    "adverse_weather_fog": (180, 180, 180),  
    "adverse_weather_snow": (220, 220, 255), 
    "animal_on_road": (150, 75, 0),      
    "debris_on_road": (100, 100, 100),   
    "overturned_vehicle": (255, 0, 0),   
    "multi_vehicle_collision": (255, 0, 0), 
    "single_vehicle_incident": (255, 0, 0), 
    "hazard_liquid_spill": (200, 150, 50), 
    "high_speed_chase": (200, 0, 200),   
    "traffic_light_malfunction": (255, 140, 0), 
    "road_construction": (255, 165, 0),  
    "post_accident_clearance": (120, 120, 120), 
    "damaged_infrastructure": (120, 120, 120), 
    "UNAVAILABLE": (80, 80, 80),      
    "ERROR_NO_MODEL_LOADED": (80, 80, 80), 
    "SYSTEM_ERROR": (80, 80, 80),     
    "INVALID_INPUT": (80, 80, 80),    
}


# --- Initialize YOLO model ---
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"[INFO] YOLOv8 model '{YOLO_MODEL_PATH}' loaded successfully.")
    print(f"[INFO] YOLOv8 loaded with {len(yolo_model.names)} classes.")
except Exception as e:
    print(f"[CRITICAL ERROR] Could not load YOLO model from {YOLO_MODEL_PATH}: {e}")
    print("Please ensure 'yolov8n.pt' is in your 'models' folder or adjust YOLO_MODEL_PATH in constants.py.")
    sys.exit(1)

# --- Font for drawing text on image ---
if sys.platform == "win32":
    default_font_path = "C:/Windows/Fonts/arial.ttf" 
elif sys.platform == "darwin":
    default_font_path = "/Library/Fonts/Arial.ttf" 
else:
    default_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 

try:
    if os.path.exists(default_font_path):
        display_font_base = ImageFont.truetype(default_font_path, 20) 
        print(f"[INFO] Custom font loaded from: {default_font_path}")
    else:
        display_font_base = ImageFont.load_default()
        print(f"[WARNING] Custom font not found at {default_font_path}. Using default PIL font.")
except Exception as e:
    print(f"[WARNING] Error loading custom font: {e}. Using default PIL font.")
    display_font_base = ImageFont.load_default()

# Global deques for temporal smoothing and event log
scene_prediction_history = deque(maxlen=SCENE_SMOOTHING_WINDOW_SIZE)
event_log_history = deque(maxlen=10) 

# Global state for alert system
current_alert_level = "OBSERVATION"
consecutive_accident_frames = 0 
last_alert_timestamp = 0 

# For simulating dynamic effects
frame_counter_for_animation = 0

# --- Custom Drawing Functions for HUD elements ---

def draw_rounded_rectangle(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], radius: int, fill=None, outline=None, width=1):
    """Draw a rectangle with rounded corners."""
    draw.rounded_rectangle([xy[0], xy[1], xy[2], xy[3]], radius=radius, fill=fill, outline=outline, width=width)

def draw_hud_box(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], fill: Tuple[int, int, int, int], outline: Tuple[int, int, int, int], thickness: int, corner_radius: int):
    """Draws a solid filled box with rounded corners and an outline."""
    draw_rounded_rectangle(draw, xy, corner_radius, fill=fill, outline=outline, width=thickness)

def draw_hud_text(draw: ImageDraw.ImageDraw, text: str, position: Tuple[int, int], font: ImageFont.FreeTypeFont, text_color: Tuple[int, int, int, int]):
    """Draws text on the HUD."""
    draw.text(position, text, fill=text_color, font=font)

def draw_glowing_line(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int, int], base_width: int, glow_strength: int = 3):
    """Draws a line with a simulated glow effect."""
    for i in range(glow_strength, 0, -1):
        alpha = int(color[3] * (i / glow_strength) * 0.5) 
        draw.line((x1, y1, x2, y2), fill=color[:3] + (alpha,), width=base_width + i * 2)
    draw.line((x1, y1, x2, y2), fill=color, width=base_width)

def draw_wireframe_element(draw: ImageDraw.ImageDraw, frame_width: int, frame_height: int, line_color: Tuple[int, int, int, int], base_thickness: int, animation_frame: int):
    """Draws abstract wireframe elements with subtle animation."""
    # Scale thickness based on frame size
    scaled_thickness = max(1, int(base_thickness * (min(frame_width, frame_height) / 800.0)))

    # Base corner lines
    coords = [
        (0, 50, 50, 0), (0, 100, 100, 0), 
        (frame_width, 50, frame_width - 50, 0), (frame_width, 100, frame_width - 100, 0), 
        (0, frame_height - 50, 50, frame_height), (0, frame_height - 100, 100, frame_height), 
        (frame_width, frame_height - 50, frame_width - 50, frame_height), (frame_width, frame_height - 100, frame_width - 100, frame_height) 
    ]
    for x1, y1, x2, y2 in coords:
        # Scale coordinates for corner lines relative to frame size for true responsiveness
        scaled_x1 = int(x1 * (frame_width / 1280.0))
        scaled_y1 = int(y1 * (frame_height / 720.0))
        scaled_x2 = int(x2 * (frame_width / 1280.0))
        scaled_y2 = int(y2 * (frame_height / 720.0))

        draw_glowing_line(draw, scaled_x1, scaled_y1, scaled_x2, scaled_y2, line_color, scaled_thickness)

    # Dynamic Grid Overlay - subtle shifting
    grid_alpha = int(line_color[3] * 0.2) 
    grid_color = line_color[:3] + (grid_alpha,)
    
    # Horizontal grid lines
    num_h_lines = max(5, int(frame_height / 120)) # More lines for larger screens
    for i in range(num_h_lines):
        y_offset_base = (animation_frame % 200) * (100.0 / 200.0) # Normalized offset 0-100
        y_pos = int((i * (frame_height / num_h_lines)) + y_offset_base) % frame_height
        draw_glowing_line(draw, 0, y_pos, frame_width, y_pos, grid_color, max(1, scaled_thickness // 2))

    # Vertical grid lines
    num_v_lines = max(5, int(frame_width / 120)) # More lines for larger screens
    for i in range(num_v_lines):
        x_offset_base = (animation_frame % 200) * (100.0 / 200.0) # Normalized offset 0-100
        x_pos = int((i * (frame_width / num_v_lines)) + x_offset_base) % frame_width
        draw_glowing_line(draw, x_pos, 0, x_pos, frame_height, grid_color, max(1, scaled_thickness // 2))

    # Center-out pulsating circle (simulated)
    pulse_radius_max = min(frame_width, frame_height) // 4
    pulse_radius = int(pulse_radius_max * (math.sin(animation_frame * 0.05) * 0.5 + 0.5)) 
    pulse_alpha = int(line_color[3] * (1 - (pulse_radius / pulse_radius_max if pulse_radius_max > 0 else 0)) * 0.7) 
    pulse_color = line_color[:3] + (pulse_alpha,)
    draw.ellipse([frame_width//2 - pulse_radius, frame_height//2 - pulse_radius, 
                  frame_width//2 + pulse_radius, frame_height//2 + pulse_radius], 
                 outline=pulse_color, width=max(1, scaled_thickness))
    
def draw_bar_graph(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, values: List[Dict[str, Any]], font: ImageFont.FreeTypeFont, base_color: Tuple[int,int,int,int]):
    """Draws a simple bar graph for confidences."""
    bar_spacing = max(2, int(width * 0.01)) # Scale bar spacing
    dummy_text_bbox = font.getbbox("Tg")
    font_height = dummy_text_bbox[3] - dummy_text_bbox[1]

    max_bar_height = height - font_height * 2 - bar_spacing * 2
    num_bars = len(values)
    if num_bars == 0: return

    individual_bar_width = (width - (num_bars + 1) * bar_spacing) // num_bars
    if individual_bar_width <= 0: return 

    for i, pred_dict in enumerate(values):
        label = pred_dict["label"]
        value = pred_dict["confidence"]

        bar_height_actual = int(max_bar_height * value)
        bar_x1 = x + bar_spacing + i * (individual_bar_width + bar_spacing)
        bar_y1 = y + height - bar_height_actual - font_height - bar_spacing
        bar_x2 = bar_x1 + individual_bar_width
        bar_y2 = y + height - font_height - bar_spacing

        bar_color = SCENE_LABEL_COLORS.get(label, base_color)
        draw_rounded_rectangle(draw, (bar_x1, bar_y1, bar_x2, bar_y2), max(2, int(width * 0.02)), fill=bar_color+(150,), outline=bar_color, width=1) # Scale corner radius
        
        draw_hud_text(draw, f"{value:.2f}", (bar_x1, bar_y1 - font_height - 2), font, HUD_TEXT_COLOR_PRIMARY)
        draw_hud_text(draw, label.replace('_', ' ')[:8], (bar_x1, bar_y2 + 2), font, HUD_TEXT_COLOR_SECONDARY)

def draw_info_panel(draw: ImageDraw.ImageDraw, start_x: int, start_y: int, panel_width: int, panel_height: int, title: str, content_lines: List[str], font_title: ImageFont.FreeTypeFont, font_content: ImageFont.FreeTypeFont, color_primary: Tuple[int,int,int,int], color_secondary: Tuple[int,int,int,int], text_color: Tuple[int,int,int], frame_base_dim: Tuple[int, int]):
    """Draws a structured info panel with title and content lines, with scaled elements."""
    # Scale HUD elements based on current frame size relative to a base resolution (e.g., 1280x720)
    width_ratio = panel_width / frame_base_dim[0]
    height_ratio = panel_height / frame_base_dim[1]
    scale_factor = min(width_ratio, height_ratio) # Use the smaller ratio to ensure fit

    scaled_corner_radius = max(5, int(HUD_CORNER_RADIUS_BASE * scale_factor * 2)) # Adjust factor for visual appeal
    scaled_outline_width = max(1, int(HUD_OUTLINE_WIDTH_BASE * scale_factor * 1.5)) # Adjust factor for visual appeal

    draw_hud_box(draw, (start_x, start_y, start_x + panel_width, start_y + panel_height), color_primary, color_secondary, scaled_outline_width, scaled_corner_radius)
    
    title_bbox = font_title.getbbox(title)
    title_width = title_bbox[2] - title_bbox[0]
    draw_hud_text(draw, title, (start_x + (panel_width - title_width) // 2, start_y + int(10 * scale_factor)), font_title, text_color) # Scale vertical offset
    
    line_y_pos = start_y + int(10 * scale_factor) + (title_bbox[3] - title_bbox[1]) + int(10 * scale_factor)
    draw_glowing_line(draw, start_x + int(20 * scale_factor), line_y_pos, start_x + panel_width - int(20 * scale_factor), line_y_pos, color_secondary, base_width=max(1, int(1 * scale_factor)))

    content_y = start_y + int(10 * scale_factor) + (title_bbox[3] - title_bbox[1]) + int(20 * scale_factor)
    content_line_height = font_content.getbbox("Tg")[3] - font_content.getbbox("Tg")[1] + int(5 * scale_factor) # Scale line spacing

    for line in content_lines:
        draw_hud_text(draw, line, (start_x + int(20 * scale_factor), content_y), font_content, text_color)
        content_y += content_line_height


def run_traffic_monitoring():
    global current_alert_level, consecutive_accident_frames, last_alert_timestamp, frame_counter_for_animation

    cap = None
    if VIDEO_INPUT_SOURCE == "WEBCAM":
        cap = cv2.VideoCapture(0)
    elif os.path.exists(VIDEO_INPUT_SOURCE):
        cap = cv2.VideoCapture(VIDEO_INPUT_SOURCE)
    else:
        print(f"[ERROR] Video input source not found: {VIDEO_INPUT_SOURCE}")
        print("Please check VIDEO_INPUT_SOURCE in main.py or provide a valid path/option.")
        return

    if not cap.isOpened():
        print("[CRITICAL ERROR] Could not open video source.")
        return

    frame_count = 0
    start_time = time.time()
    
    # Get actual frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define a base resolution for scaling UI elements. 1280x720 is a common HD.
    BASE_UI_WIDTH = 1280
    BASE_UI_HEIGHT = 720

    # Calculate scaling factors for UI elements
    # Use the minimum of width/height ratio to ensure elements don't go off screen
    scale_factor_w = frame_width / BASE_UI_WIDTH
    scale_factor_h = frame_height / BASE_UI_HEIGHT
    global_ui_scale = min(scale_factor_w, scale_factor_h)

    # Dynamically adjust font sizes for HUD elements based on resolution
    # Font sizes should scale proportionally to the global UI scale
    hud_font_size_main_scaled = max(12, int(28 * global_ui_scale)) # Base 28
    hud_font_size_sub_scaled = max(10, int(20 * global_ui_scale)) # Base 20
    hud_font_size_small_scaled = max(8, int(14 * global_ui_scale)) # Base 14

    font_main = ImageFont.truetype(default_font_path, hud_font_size_main_scaled) if os.path.exists(default_font_path) else ImageFont.load_default()
    font_sub = ImageFont.truetype(default_font_path, hud_font_size_sub_scaled) if os.path.exists(default_font_path) else ImageFont.load_default()
    font_small = ImageFont.truetype(default_font_path, hud_font_size_small_scaled) if os.path.exists(default_font_path) else ImageFont.load_default()

    font_main_height = font_main.getbbox("Tg")[3] - font_main.getbbox("Tg")[1]
    font_sub_height = font_sub.getbbox("Tg")[3] - font_sub.getbbox("Tg")[1]
    font_small_height = font_small.getbbox("Tg")[3] - font_small.getbbox("Tg")[1]


    while True:
        ret, frame = cap.read()
        if not ret:
            print("[JARVIS-LOG] End of video or stream disconnected. Initiating shutdown sequence.")
            break

        frame_count += 1
        frame_counter_for_animation += 1

        if frame_count % FRAME_PROCESS_INTERVAL != 0:
            continue 

        hud_layer = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(hud_layer)

        # --- Draw Core Wireframe Overlay (Behind other elements, with animation) ---
        draw_wireframe_element(draw, frame_width, frame_height, HUD_BLUE_LIGHT, base_thickness=2, animation_frame=frame_counter_for_animation) # Increased base_thickness for wireframe


        # --- JARVIS-Level Scene Classification & Temporal Smoothing ---
        pil_frame_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scene_report = get_scene_prediction(pil_frame_rgb)
        current_scene_label = scene_report["main_prediction"]
        scene_confidence = scene_report["main_confidence"]
        top_predictions_for_graph = scene_report["all_predictions"]

        if scene_confidence >= MIN_SCENE_CONFIDENCE_DISPLAY:
            scene_prediction_history.append(current_scene_label)

        most_common_scene = "N/A"
        smoothed_scene_confidence = 0.0
        if scene_prediction_history:
            label_counts = {}
            for label in scene_prediction_history:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            if label_counts:
                most_common_scene = max(label_counts, key=label_counts.get)
                smoothed_scene_confidence = label_counts[most_common_scene] / len(scene_prediction_history)


        # --- Alert Level Logic & Smart Accident Filter ---
        current_cars = 0
        current_accident_impact_objects = 0
        yolo_results_for_logic = yolo_model(pil_frame_rgb, conf=YOLO_DETECTION_CONFIDENCE_OVERRIDE, verbose=False)
        for r in yolo_results_for_logic:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls[0])]
                if label in ["car", "truck", "bus"]: 
                    current_cars += 1
                if label in ACCIDENT_IMPACT_OBJECTS and float(box.conf[0]) > 0.6: 
                    current_accident_impact_objects += 1

        force_observation = False
        if most_common_scene == "dense_traffic" and smoothed_scene_confidence > 0.8:
            force_observation = True
            if current_alert_level != "OBSERVATION":
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Override: Dense Traffic Confirmed.")
        elif most_common_scene == "accident" and current_cars > DENSE_TRAFFIC_CAR_COUNT_THRESHOLD_FOR_FALSE_POSITIVE and current_accident_impact_objects < MIN_IMPACT_OBJECTS_FOR_ACCIDENT:
            force_observation = True
            if current_alert_level != "OBSERVATION":
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Anomaly: Accident-like, but dense traffic. Verification needed.")
        
        if current_alert_level != "OBSERVATION" and \
           (most_common_scene != "accident" or force_observation) and \
           (time.time() - last_alert_timestamp) > ALERT_COOLDOWN_SECONDS:
            current_alert_level = "OBSERVATION"
            consecutive_accident_frames = 0
            event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Status: Monitoring (Resolved/Cleared)")
        
        if not force_observation and most_common_scene == "accident" and smoothed_scene_confidence >= ACCIDENT_CONFIDENCE_THRESHOLD_OBSERVE:
            consecutive_accident_frames += 1
            if consecutive_accident_frames >= ACCIDENT_PERSISTENCE_FRAMES_CRITICAL and smoothed_scene_confidence >= ACCIDENT_CONFIDENCE_THRESHOLD_CRITICAL_ALERT:
                if current_alert_level != "ALERT_SENT": 
                    current_alert_level = "CRITICAL_ALERT"
                    last_alert_timestamp = time.time() 
                    print(f"[JARVIS-ALERT] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - CRITICAL ACCIDENT DETECTED! Triggering high-priority alert system.")
                    event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - CRITICAL ACCIDENT! (Conf: {smoothed_scene_confidence:.2f})")
            elif consecutive_accident_frames >= ACCIDENT_PERSISTENCE_FRAMES_WARN and smoothed_scene_confidence >= ACCIDENT_CONFIDENCE_THRESHOLD_WARN:
                current_alert_level = "WARNING"
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Potential Incident. (Conf: {smoothed_scene_confidence:.2f})")
            else:
                current_alert_level = "OBSERVATION"
        elif force_observation: 
            current_alert_level = "OBSERVATION"
            consecutive_accident_frames = 0
        else: 
            consecutive_accident_frames = 0
            current_alert_level = "OBSERVATION" 
            

        display_action_message = scene_report["suggested_action"]
        if current_alert_level == "WARNING":
            display_action_message = f"VERIFICATION REQUIRED: Potential Incident. (Conf: {smoothed_scene_confidence:.2f})"
        elif current_alert_level == "CRITICAL_ALERT":
            display_action_message = f"URGENT: DISPATCHING EMERGENCY SERVICES! Conf: {smoothed_scene_confidence:.2f}"
            if last_alert_timestamp and (time.time() - last_alert_timestamp < 5): 
                pass 
            else:
                print(f"[JARVIS-ALERT] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - CRITICAL ACCIDENT DETECTED! Alerting emergency services for scene: {most_common_scene.upper()} (Confidence: {smoothed_scene_confidence:.2f})")
                last_alert_timestamp = time.time()
        elif current_alert_level == "ALERT_SENT":
            display_action_message = "ALERT DISPATCHED. Monitoring Scene for Updates."
            
        
        # --- Draw Main HUD Elements ---
        # Panel sizes and positions are now relative to frame_width/height
        padding = int(20 * global_ui_scale) # Scale padding
        hud_outline_width = max(1, int(HUD_OUTLINE_WIDTH_BASE * global_ui_scale))
        hud_corner_radius = max(5, int(HUD_CORNER_RADIUS_BASE * global_ui_scale))

        # Top-left HUD block: Scene, Status, and Action
        # Panel width will be a proportion of frame_width, height proportion of frame_height
        panel1_width = int(frame_width * 0.30) # ~30% of width
        panel1_height = int(frame_height * 0.25) # ~25% of height
        panel1_x = padding
        panel1_y = padding

        current_y_in_panel1 = panel1_y + int(10 * global_ui_scale) # Start inner content below panel top margin

        # Scene Status (Main display) with dynamic color/glow
        scene_color_for_display = SCENE_LABEL_COLORS.get(most_common_scene, HUD_TEXT_COLOR_PRIMARY)
        scene_display_text = f"SCENE: {most_common_scene.replace('_', ' ').upper()}"
        
        pulse_alpha = int(255 * (math.sin(frame_counter_for_animation * 0.1) * 0.2 + 0.8)) 
        text_color_pulsating = scene_color_for_display[:3] + (pulse_alpha,)

        draw_hud_text(draw, scene_display_text, (panel1_x + int(20 * global_ui_scale), current_y_in_panel1), font_main, text_color_pulsating)
        current_y_in_panel1 += font_main_height + int(5 * global_ui_scale)
        draw_hud_text(draw, f"CONF: {smoothed_scene_confidence:.2f}", (panel1_x + int(20 * global_ui_scale), current_y_in_panel1), font_sub, HUD_TEXT_COLOR_HIGHLIGHT)
        
        # Underline/Separator with glow and animated scanline
        line_start_x = panel1_x + int(15 * global_ui_scale)
        line_end_x = panel1_x + panel1_width - int(15 * global_ui_scale)
        line_y = current_y_in_panel1 + font_sub_height + int(15 * global_ui_scale)
        draw_glowing_line(draw, line_start_x, line_y, line_end_x, line_y, HUD_CYAN_LIGHT, base_width=max(1, int(2 * global_ui_scale)))
        
        scan_x = line_start_x + int((line_end_x - line_start_x) * (frame_counter_for_animation % 60 / 60.0))
        draw_glowing_line(draw, scan_x, line_y - int(5 * global_ui_scale), scan_x, line_y + int(5 * global_ui_scale), HUD_CYAN_LIGHT, base_width=max(1, int(1 * global_ui_scale)))

        current_y_in_panel1 = line_y + int(10 * global_ui_scale)

        # Alert Level & Action Status
        alert_text_color = HUD_TEXT_COLOR_PRIMARY
        alert_bg_color = HUD_BLUE_DARK
        alert_outline_color = HUD_BLUE_LIGHT
        
        if current_alert_level == "WARNING":
            alert_text_color = SCENE_LABEL_COLORS["dense_traffic"] 
            alert_bg_color = HUD_YELLOW_ACCENT
            alert_outline_color = HUD_YELLOW_ACCENT
        elif current_alert_level in ["CRITICAL_ALERT", "ALERT_SENT"]:
            alert_text_color = SCENE_LABEL_COLORS["accident"] 
            alert_bg_color = HUD_RED_CRITICAL
            alert_outline_color = HUD_RED_CRITICAL
            pulsating_fill_alpha = int(100 * (math.sin(frame_counter_for_animation * 0.3) * 0.5 + 0.5))
            pulsating_fill_color = (HUD_RED_CRITICAL[0], HUD_RED_CRITICAL[1], HUD_RED_CRITICAL[2], pulsating_fill_alpha)
            
            draw.rectangle((0, 0, frame_width-1, frame_height-1), outline=HUD_RED_CRITICAL, width=max(2, int(frame_width * 0.005)), 
                           fill=pulsating_fill_color)
                           

        draw_hud_text(draw, f"STATUS: {current_alert_level.replace('_', ' ').upper()}", (panel1_x + int(20 * global_ui_scale), current_y_in_panel1), font_sub, alert_text_color)
        current_y_in_panel1 += font_sub_height + int(5 * global_ui_scale)
        draw_hud_text(draw, display_action_message, (panel1_x + int(20 * global_ui_scale), current_y_in_panel1), font_small, HUD_TEXT_COLOR_SECONDARY)
        
        # Draw the overall panel last so it overlays the content appropriately.
        draw_hud_box(draw, (panel1_x, panel1_y, panel1_x + panel1_width, panel1_y + panel1_height), HUD_BLUE_DARK, HUD_BLUE_LIGHT, hud_outline_width, hud_corner_radius)

        # Re-draw the texts to be on top of the box border
        draw_hud_text(draw, scene_display_text, (panel1_x + int(20 * global_ui_scale), panel1_y + int(10 * global_ui_scale)), font_main, text_color_pulsating)
        draw_hud_text(draw, f"CONF: {smoothed_scene_confidence:.2f}", (panel1_x + int(20 * global_ui_scale), panel1_y + int(10 * global_ui_scale) + font_main_height + int(5 * global_ui_scale)), font_sub, HUD_TEXT_COLOR_HIGHLIGHT)
        
        # Re-draw alert texts
        current_y_for_alert_redraw = panel1_y + int(10 * global_ui_scale) + (font_main_height * 2) + int(30 * global_ui_scale) + int(10 * global_ui_scale) # Adjusted
        draw_hud_text(draw, f"STATUS: {current_alert_level.replace('_', ' ').upper()}", (panel1_x + int(20 * global_ui_scale), current_y_for_alert_redraw), font_sub, alert_text_color)
        draw_hud_text(draw, display_action_message, (panel1_x + int(20 * global_ui_scale), current_y_for_alert_redraw + font_sub_height + int(5 * global_ui_scale)), font_small, HUD_TEXT_COLOR_SECONDARY)


        # --- Object Detection Panel (Top Right) ---
        panel2_width = int(frame_width * 0.35) # A bit wider
        panel2_height = int(frame_height * 0.35) # Taller
        panel2_x = frame_width - panel2_width - padding
        panel2_y = padding

        # Draw panel background first
        draw_hud_box(draw, (panel2_x, panel2_y, panel2_x + panel2_width, panel2_y + panel2_height), HUD_BLUE_DARK, HUD_CYAN_LIGHT, hud_outline_width, hud_corner_radius)
        draw_hud_text(draw, "OBJECT CLASSIFICATION", (panel2_x + int(20 * global_ui_scale), panel2_y + int(15 * global_ui_scale)), font_sub, HUD_TEXT_COLOR_PRIMARY)
        draw_glowing_line(draw, panel2_x + int(20 * global_ui_scale), panel2_y + int(50 * global_ui_scale), panel2_x + panel2_width - int(20 * global_ui_scale), panel2_y + int(50 * global_ui_scale), HUD_CYAN_LIGHT, base_width=max(1, int(1 * global_ui_scale)))

        object_counts: Dict[str, int] = {}
        for r in yolo_results_for_logic: 
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]

                if label in YOLO_CLASSES:
                    object_counts[label] = object_counts.get(label, 0) + 1
                    
                    color = BBOX_COLORS.get(label, (200, 200, 200))
                    color_with_alpha = color + (150,) 
                    
                    draw.rectangle([x1, y1, x2, y2], outline=color_with_alpha, width=max(1, int(frame_width * 0.0015 * global_ui_scale))) # Scale bbox width
                    
                    text_label = f"{label.upper()} ({conf:.2f})"
                    bbox_text = font_small.getbbox(text_label)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                    
                    text_x = x1 + max(1, int(4 * global_ui_scale))
                    text_y = y1 - text_height - max(1, int(6 * global_ui_scale))
                    if text_y < 0: text_y = y1 + max(1, int(2 * global_ui_scale)) 
                        
                    draw_rounded_rectangle(draw, [text_x - max(1, int(2 * global_ui_scale)), text_y - max(1, int(2 * global_ui_scale)), text_x + text_width + max(1, int(4 * global_ui_scale)), text_y + text_height + max(1, int(4 * global_ui_scale))], radius=max(1, int(4 * global_ui_scale)), fill=color_with_alpha)
                    draw.text((text_x, text_y), text_label, fill=(0,0,0), font=font_small)

        obj_content_y_start = panel2_y + int(60 * global_ui_scale)
        obj_line_height = font_small_height + int(5 * global_ui_scale)
        max_lines_obj = (panel2_height - int(60 * global_ui_scale)) // obj_line_height
        
        current_obj_lines_count = 0
        for obj_label, count in object_counts.items():
            if current_obj_lines_count < max_lines_obj:
                draw_hud_text(draw, f"- {obj_label.capitalize()}: {count}", (panel2_x + int(20 * global_ui_scale), obj_content_y_start + current_obj_lines_count * obj_line_height), font_small, HUD_TEXT_COLOR_SECONDARY)
                current_obj_lines_count += 1
        
        total_detected_objects = sum(object_counts.values())
        if current_obj_lines_count < max_lines_obj:
             draw_hud_text(draw, f"TOTAL: {total_detected_objects}", (panel2_x + int(20 * global_ui_scale), obj_content_y_start + current_obj_lines_count * obj_line_height), font_small, HUD_TEXT_COLOR_HIGHLIGHT)


        # --- Scene Confidence Bar Graph (Bottom of Top Right Panel) ---
        # Position relative to panel2's bottom, not absolute frame bottom
        graph_height = int(frame_height * 0.1) # Proportion of frame height
        graph_y_start = panel2_y + panel2_height - graph_height - int(10 * global_ui_scale) # Adjusted position
        
        draw_hud_text(draw, "SCENE CONFIDENCE:", (panel2_x + int(20 * global_ui_scale), graph_y_start - font_small_height - int(5 * global_ui_scale)), font_small, HUD_TEXT_COLOR_PRIMARY)
        draw_bar_graph(draw, panel2_x + int(10 * global_ui_scale), graph_y_start, panel2_width - int(20 * global_ui_scale), graph_height, top_predictions_for_graph, font_small, HUD_CYAN_LIGHT)


        # --- Event Log Panel (Bottom Left) ---
        panel3_width = int(frame_width * 0.30) 
        panel3_height = int(frame_height * 0.25)
        panel3_x = padding
        panel3_y = frame_height - panel3_height - padding
        
        draw_hud_box(draw, (panel3_x, panel3_y, panel3_x + panel3_width, panel3_y + panel3_height), HUD_BLUE_DARK, HUD_BLUE_LIGHT, hud_outline_width, hud_corner_radius)
        draw_hud_text(draw, "EVENT LOG", (panel3_x + int(20 * global_ui_scale), panel3_y + int(15 * global_ui_scale)), font_sub, HUD_TEXT_COLOR_PRIMARY)
        draw_glowing_line(draw, panel3_x + int(20 * global_ui_scale), panel3_y + int(50 * global_ui_scale), panel3_x + panel3_width - int(20 * global_ui_scale), panel3_y + int(50 * global_ui_scale), HUD_BLUE_LIGHT, base_width=max(1, int(1 * global_ui_scale)))

        log_content_y_start = panel3_y + int(60 * global_ui_scale)
        log_line_height = font_small_height + int(5 * global_ui_scale) 
        max_log_lines = (panel3_height - int(60 * global_ui_scale)) // log_line_height

        if len(event_log_history) > 0:
            effective_log_length = len(event_log_history)
            scroll_duration_frames = effective_log_length * 30 # Slower scroll based on number of lines
            
            if scroll_duration_frames == 0: scroll_denominator = 1 
            else: scroll_denominator = scroll_duration_frames

            log_scroll_pos = (frame_counter_for_animation % scroll_denominator) / scroll_denominator # Normalized 0.0 to 1.0
            
            start_log_index_float = (len(event_log_history) - max_log_lines) * log_scroll_pos
            start_log_index_int = int(start_log_index_float)
            fractional_offset_for_smooth_scroll = start_log_index_float - start_log_index_int
            
            for i in range(max_log_lines):
                actual_log_index = start_log_index_int + i
                
                if actual_log_index < len(event_log_history) and actual_log_index >= 0:
                    log_entry = event_log_history[actual_log_index]
                    y_pos = log_content_y_start + i * log_line_height - fractional_offset_for_smooth_scroll * log_line_height
                    draw_hud_text(draw, log_entry, (panel3_x + int(20 * global_ui_scale), y_pos), font_small, HUD_TEXT_COLOR_SECONDARY)
        else: 
            draw_hud_text(draw, "No events to display.", (panel3_x + int(20 * global_ui_scale), log_content_y_start), font_small, HUD_TEXT_COLOR_SECONDARY)


        # --- System Status Panel (Bottom Right) ---
        panel4_width = int(frame_width * 0.25) # Smaller width
        panel4_height = int(frame_height * 0.25) # Same height
        panel4_x = frame_width - panel4_width - padding
        panel4_y = frame_height - panel4_height - padding

        draw_hud_box(draw, (panel4_x, panel4_y, panel4_x + panel4_width, panel4_y + panel4_height), HUD_BLUE_DARK, HUD_GREEN_LIGHT, hud_outline_width, hud_corner_radius)
        draw_hud_text(draw, "SYSTEM HEALTH", (panel4_x + int(20 * global_ui_scale), panel4_y + int(15 * global_ui_scale)), font_sub, HUD_TEXT_COLOR_PRIMARY)
        draw_glowing_line(draw, panel4_x + int(20 * global_ui_scale), panel4_y + int(50 * global_ui_scale), panel4_x + panel4_width - int(20 * global_ui_scale), panel4_y + int(50 * global_ui_scale), HUD_GREEN_LIGHT, base_width=max(1, int(1 * global_ui_scale)))

        fps_text = ""
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            fps_text = f"{fps:.1f}"
        
        simulated_cpu_load = 50 + 20 * math.sin(frame_counter_for_animation * 0.05) 
        simulated_gpu_load = 60 + 15 * math.cos(frame_counter_for_animation * 0.07) 
        simulated_data_rate = 10 + 5 * math.sin(frame_counter_for_animation * 0.03)

        sys_lines = [
            f"Frames: {frame_count}",
            f"FPS: {fps_text}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            f"CPU Load: {simulated_cpu_load:.1f}%",
            f"GPU Load: {simulated_gpu_load:.1f}%",
            f"Data Rate: {simulated_data_rate:.1f} MB/s",
            f"Device: {str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).upper()}"
        ]
        
        sys_content_y_start = panel4_y + int(60 * global_ui_scale)
        sys_line_height = font_small_height + int(5 * global_ui_scale) 
        for i, line in enumerate(sys_lines):
            draw_hud_text(draw, line, (panel4_x + int(20 * global_ui_scale), sys_content_y_start + i * sys_line_height), font_small, HUD_TEXT_COLOR_SECONDARY)


        # --- Composite the HUD layer onto the original frame ---
        frame_rgba = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        pil_combined_image = Image.alpha_composite(frame_rgba, hud_layer)

        # Convert back to OpenCV format (BGR) for display
        frame_processed_cv2 = cv2.cvtColor(np.array(pil_combined_image), cv2.COLOR_RGBA2BGR)

        cv2.imshow("AI-Powered Traffic Monitoring (JARVIS-Level HUD)", frame_processed_cv2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[JARVIS-LOG] Traffic monitoring terminated. All systems offline.")


if __name__ == "__main__":
    run_traffic_monitoring()
