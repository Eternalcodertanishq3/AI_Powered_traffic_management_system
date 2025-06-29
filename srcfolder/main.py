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
import random # For simulated dynamism

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
SCENE_SMOOTHING_WINDOW_SIZE = 20 # Increased window for even smoother scene prediction
MIN_SCENE_CONFIDENCE_DISPLAY = 0.50 

# ACCIDENT ALERT THRESHOLDS (Increased for more caution)
ACCIDENT_CONFIDENCE_THRESHOLD_OBSERVE = 0.70 # Higher for initial observe
ACCIDENT_CONFIDENCE_THRESHOLD_WARN = 0.90 # Higher for warning
ACCIDENT_CONFIDENCE_THRESHOLD_CRITICAL_ALERT = 0.95 # Very high for critical
ACCIDENT_PERSISTENCE_FRAMES_WARN = 10 # Needs to be accident for 10 consecutive frames for Warning
ACCIDENT_PERSISTENCE_FRAMES_CRITICAL = 15 # Needs to be accident for 15 consecutive frames for Critical Alert
ALERT_COOLDOWN_SECONDS = 30 # Longer cooldown to prevent alert spam

# Heuristic for reducing false positives for accident in dense traffic
DENSE_TRAFFIC_CAR_COUNT_THRESHOLD_FOR_FALSE_POSITIVE = 15 # If more than X cars AND no specific accident indicators, be VERY cautious
ACCIDENT_IMPACT_OBJECTS = ["stalled_vehicle", "debris_on_road", "overturned_vehicle", "fire", "truck", "bus"] # Consider large vehicles as "impact" objects if they are stationary/part of incident
MIN_IMPACT_OBJECTS_FOR_ACCIDENT = 1 # At least one high-confidence impact object needed to confirm accident in dense traffic


# --- YOLO DETECTION CONFIG ---
YOLO_DETECTION_CONFIDENCE_OVERRIDE = 0.25 # Slightly lower for more detections, as we'll filter them later

# --- UI Styling (Iron Man / JARVIS Vibe) ---
# RGBA colors for transparency
HUD_BLUE_DARK = (10, 20, 50, 220)       # Dark background for modules
HUD_BLUE_MEDIUM = (20, 60, 100, 180)    # Slightly lighter for some elements
HUD_BLUE_LIGHT = (30, 144, 255, 180)    # Dodger Blue for main accents
HUD_CYAN_LIGHT = (0, 255, 255, 180)     # Cyan for highlights/wireframes
HUD_GREEN_LIGHT = (0, 255, 127, 180)    # Spring Green for safe/normal status
HUD_YELLOW_ACCENT = (255, 255, 0, 180)  # Yellow for warnings
HUD_RED_CRITICAL = (255, 69, 0, 180)    # OrangeRed for critical alerts
HUD_TEXT_COLOR_PRIMARY = (220, 240, 255, 255) # Light blue-white for main text
HUD_TEXT_COLOR_SECONDARY = (180, 200, 255, 255) # Slightly darker for sub-text
HUD_TEXT_COLOR_HIGHLIGHT = (0, 255, 255, 255) # Pure cyan for key values
HUD_OUTLINE_WIDTH = 2 # General outline width for HUD elements
HUD_CORNER_RADIUS = 15 # General corner radius for HUD elements

# Specific colors for scene labels (RGB tuples for PIL)
SCENE_LABEL_COLORS = {
    "normal_traffic": (0, 255, 100),       # Bright Green
    "dense_traffic": (255, 200, 0),      # Orange-Yellow
    "sparse_traffic": (0, 150, 255),     # Sky Blue
    "accident": (255, 50, 50),             # Vivid Red
    "emergency_vehicle_passing": (255, 100, 0), # Deep Orange
    "road_block": (255, 80, 0),          # Bright Orange-Red
    "stalled_vehicle": (255, 150, 0),    # Golden Orange
    "pedestrian_crossing": (100, 255, 255), # Light Cyan
    "fire": (255, 0, 0),                 # Pure Red
    "adverse_weather_rain": (100, 180, 255), # Light Steel Blue
    "adverse_weather_fog": (180, 180, 180),  # Light Gray
    "adverse_weather_snow": (220, 220, 255), # Lavender
    "animal_on_road": (150, 75, 0),      # Brown
    "debris_on_road": (100, 100, 100),   # Dark Gray
    "overturned_vehicle": (255, 0, 0),   # Red
    "multi_vehicle_collision": (255, 0, 0), # Red
    "single_vehicle_incident": (255, 0, 0), # Red
    "hazard_liquid_spill": (200, 150, 50), # Goldenrod
    "high_speed_chase": (200, 0, 200),   # Purple
    "traffic_light_malfunction": (255, 140, 0), # Dark Orange
    "road_construction": (255, 165, 0),  # Orange
    "post_accident_clearance": (120, 120, 120), # Medium Gray
    "damaged_infrastructure": (120, 120, 120), # Medium Gray
    "UNAVAILABLE": (80, 80, 80),      # Dark Gray for errors
    "ERROR_NO_MODEL_LOADED": (80, 80, 80), # Dark Gray for errors
    "SYSTEM_ERROR": (80, 80, 80),     # Dark Gray for errors
    "INVALID_INPUT": (80, 80, 80),    # Dark Gray for errors
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
event_log_history = deque(maxlen=10) # Store last 10 events

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
    # Base corner lines
    coords = [
        (0, 50, 50, 0), (0, 100, 100, 0), # Top-left
        (frame_width, 50, frame_width - 50, 0), (frame_width, 100, frame_width - 100, 0), # Top-right
        (0, frame_height - 50, 50, frame_height), (0, frame_height - 100, 100, frame_height), # Bottom-left
        (frame_width, frame_height - 50, frame_width - 50, frame_height), (frame_width, frame_height - 100, frame_width - 100, frame_height) # Bottom-right
    ]
    for x1, y1, x2, y2 in coords:
        draw_glowing_line(draw, x1, y1, x2, y2, line_color, base_thickness)

    # Dynamic Grid Overlay - subtle shifting
    grid_alpha = int(line_color[3] * 0.2) # Very subtle
    grid_color = line_color[:3] + (grid_alpha,)
    
    # Horizontal grid lines
    num_h_lines = int(frame_height / 100)
    for i in range(num_h_lines):
        y_offset = (animation_frame % 200) * (frame_height / 200.0) # Vertical scroll effect
        y_pos = int(i * 100 + y_offset - frame_height / 2) % frame_height
        if y_pos < 0: y_pos += frame_height # Wrap around
        draw_glowing_line(draw, 0, y_pos, frame_width, y_pos, grid_color, 1)

    # Vertical grid lines
    num_v_lines = int(frame_width / 100)
    for i in range(num_v_lines):
        x_offset = (animation_frame % 200) * (frame_width / 200.0) # Horizontal scroll effect
        x_pos = int(i * 100 + x_offset - frame_width / 2) % frame_width
        if x_pos < 0: x_pos += frame_width # Wrap around
        draw_glowing_line(draw, x_pos, 0, x_pos, frame_height, grid_color, 1)

    # Center-out pulsating circle (simulated)
    pulse_radius_max = min(frame_width, frame_height) // 4
    pulse_radius = int(pulse_radius_max * (math.sin(animation_frame * 0.05) * 0.5 + 0.5)) # 0 to max_radius
    pulse_alpha = int(line_color[3] * (1 - pulse_radius / pulse_radius_max) * 0.7) if pulse_radius_max > 0 else 0
    pulse_color = line_color[:3] + (pulse_alpha,)
    draw.ellipse([frame_width//2 - pulse_radius, frame_height//2 - pulse_radius, 
                  frame_width//2 + pulse_radius, frame_height//2 + pulse_radius], 
                 outline=pulse_color, width=2)
    
def draw_bar_graph(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, values: List[Dict[str, Any]], font: ImageFont.FreeTypeFont, base_color: Tuple[int,int,int,int]):
    """Draws a simple bar graph for confidences."""
    bar_spacing = 5
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
        draw_rounded_rectangle(draw, (bar_x1, bar_y1, bar_x2, bar_y2), 5, fill=bar_color+(150,), outline=bar_color, width=1)
        
        draw_hud_text(draw, f"{value:.2f}", (bar_x1, bar_y1 - font_height - 2), font, HUD_TEXT_COLOR_PRIMARY)
        draw_hud_text(draw, label.replace('_', ' ')[:8], (bar_x1, bar_y2 + 2), font, HUD_TEXT_COLOR_SECONDARY)

def draw_info_panel(draw: ImageDraw.ImageDraw, start_x: int, start_y: int, panel_width: int, panel_height: int, title: str, content_lines: List[str], font_title: ImageFont.FreeTypeFont, font_content: ImageFont.FreeTypeFont, color_primary: Tuple[int,int,int,int], color_secondary: Tuple[int,int,int,int], text_color: Tuple[int,int,int]):
    """Draws a structured info panel with title and content lines."""
    panel_radius = HUD_CORNER_RADIUS
    draw_hud_box(draw, (start_x, start_y, start_x + panel_width, start_y + panel_height), color_primary, color_secondary, HUD_OUTLINE_WIDTH, panel_radius)
    
    title_bbox = font_title.getbbox(title)
    title_width = title_bbox[2] - title_bbox[0]
    draw_hud_text(draw, title, (start_x + (panel_width - title_width) // 2, start_y + 10), font_title, text_color)
    
    draw_glowing_line(draw, start_x + 20, start_y + 10 + (title_bbox[3] - title_bbox[1]) + 10, start_x + panel_width - 20, start_y + 10 + (title_bbox[3] - title_bbox[1]) + 10, color_secondary, base_width=1)

    content_y = start_y + 10 + (title_bbox[3] - title_bbox[1]) + 20
    for line in content_lines:
        draw_hud_text(draw, line, (start_x + 20, content_y), font_content, text_color)
        content_y += font_content.getbbox("Tg")[3] - font_content.getbbox("Tg")[1] + 5 # Line spacing


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
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    hud_font_size_main_scaled = max(18, int(frame_height / 30))
    hud_font_size_sub_scaled = max(14, int(frame_height / 40))
    hud_font_size_small_scaled = max(10, int(frame_height / 55))

    font_main = ImageFont.truetype(default_font_path, hud_font_size_main_scaled) if os.path.exists(default_font_path) else ImageFont.load_default()
    font_sub = ImageFont.truetype(default_font_path, hud_font_size_sub_scaled) if os.path.exists(default_font_path) else ImageFont.load_default()
    font_small = ImageFont.truetype(default_font_path, hud_font_size_small_scaled) if os.path.exists(default_font_path) else ImageFont.load_default()

    # Pre-calculate font heights for consistent spacing
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

        # Create a blank transparent layer for drawing HUD elements
        hud_layer = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(hud_layer)

        # --- Draw Core Wireframe Overlay (Behind other elements, with animation) ---
        draw_wireframe_element(draw, frame_width, frame_height, HUD_BLUE_LIGHT, base_thickness=1, animation_frame=frame_counter_for_animation)


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
        # Re-run YOLO for specific object counts needed for logic. Optimize if performance is an issue.
        yolo_results_for_logic = yolo_model(pil_frame_rgb, conf=YOLO_DETECTION_CONFIDENCE_OVERRIDE, verbose=False)
        for r in yolo_results_for_logic:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls[0])]
                if label in ["car", "truck", "bus"]: # Summing all vehicle types
                    current_cars += 1
                if label in ACCIDENT_IMPACT_OBJECTS and float(box.conf[0]) > 0.6: 
                    current_accident_impact_objects += 1

        # Heuristic for reducing false positives:
        # If classified as accident but in dense traffic and few/no direct impact objects
        # OR if main scene is definitively 'dense_traffic' with high confidence.
        force_observation = False
        if most_common_scene == "dense_traffic" and smoothed_scene_confidence > 0.8: # Very high confidence in dense traffic
            force_observation = True
            if current_alert_level != "OBSERVATION":
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Override: Dense Traffic Confirmed.")
        elif most_common_scene == "accident" and current_cars > DENSE_TRAFFIC_CAR_COUNT_THRESHOLD_FOR_FALSE_POSITIVE and current_accident_impact_objects < MIN_IMPACT_OBJECTS_FOR_ACCIDENT:
            # Predicted accident, but looks like very dense, non-impacted traffic
            force_observation = True
            if current_alert_level != "OBSERVATION":
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Anomaly: Accident-like, but dense traffic. Verification needed.")
        
        # Reset alert level if scene changes significantly or long cooldown
        if current_alert_level != "OBSERVATION" and \
           (most_common_scene != "accident" or force_observation) and \
           (time.time() - last_alert_timestamp) > ALERT_COOLDOWN_SECONDS:
            current_alert_level = "OBSERVATION"
            consecutive_accident_frames = 0
            event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Status: Monitoring (Resolved/Cleared)")
        
        # Main alert escalation logic
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
        elif force_observation: # If forced to observation, ensure counters reset
            current_alert_level = "OBSERVATION"
            consecutive_accident_frames = 0
        else: # Not an accident scenario
            consecutive_accident_frames = 0
            current_alert_level = "OBSERVATION" # Explicitly set to observation if not actively detecting incident
            

        # Determine action message based on alert level
        display_action_message = scene_report["suggested_action"]
        if current_alert_level == "WARNING":
            display_action_message = f"VERIFICATION REQUIRED: Potential Incident. (Conf: {smoothed_scene_confidence:.2f})"
        elif current_alert_level == "CRITICAL_ALERT":
            display_action_message = f"URGENT: DISPATCHING EMERGENCY SERVICES! Conf: {smoothed_scene_confidence:.2f}"
            # This console print is already in the main logic block for critical alert for cooldown handling
        elif current_alert_level == "ALERT_SENT":
            display_action_message = "ALERT DISPATCHED. Monitoring Scene for Updates."
            
        
        # --- Draw Main HUD Elements ---
        current_y_pos_top = 20 
        
        scene_color_for_display = SCENE_LABEL_COLORS.get(most_common_scene, HUD_TEXT_COLOR_PRIMARY)
        scene_display_text = f"SCENE: {most_common_scene.replace('_', ' ').upper()}"
        
        pulse_alpha = int(255 * (math.sin(frame_counter_for_animation * 0.1) * 0.2 + 0.8)) 
        text_color_pulsating = scene_color_for_display[:3] + (pulse_alpha,)

        draw_hud_text(draw, scene_display_text, (30, current_y_pos_top + 10), font_main, text_color_pulsating)
        draw_hud_text(draw, f"CONF: {smoothed_scene_confidence:.2f}", (30, current_y_pos_top + 10 + font_main_height + 5), font_sub, HUD_TEXT_COLOR_HIGHLIGHT)

        text_bbox = font_main.getbbox(scene_display_text)
        line_start_x = 25
        line_end_x = line_start_x + text_bbox[2] - text_bbox[0] + 50
        line_y = current_y_pos_top + (font_main_height * 2) + 30
        draw_glowing_line(draw, line_start_x, line_y, line_end_x, line_y, HUD_CYAN_LIGHT, base_width=2)
        
        scan_x = line_start_x + int((line_end_x - line_start_x) * (frame_counter_for_animation % 60 / 60.0))
        draw_glowing_line(draw, scan_x, line_y - 5, scan_x, line_y + 5, HUD_CYAN_LIGHT, base_width=1)

        current_y_pos_top = line_y + 10


        # Alert Level & Action Status Box
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
                           


        draw_hud_box(draw, (10, current_y_pos_top + 10, 380, current_y_pos_top + 120), alert_bg_color, alert_outline_color, HUD_OUTLINE_WIDTH, HUD_CORNER_RADIUS)
        draw_hud_text(draw, f"STATUS: {current_alert_level.replace('_', ' ').upper()}", (30, current_y_pos_top + 20), font_sub, alert_text_color)
        draw_hud_text(draw, display_action_message, (30, current_y_pos_top + 55), font_small, HUD_TEXT_COLOR_PRIMARY)
        

        # --- Object Detection Panel (Top Right) ---
        panel_padding = 20
        panel_width_obj = 380 
        panel_height_obj = 240 
        panel_x_obj = frame_width - panel_width_obj - panel_padding
        panel_y_obj = panel_padding

        draw_hud_box(draw, (panel_x_obj, panel_y_obj, panel_x_obj + panel_width_obj, panel_y_obj + panel_height_obj), HUD_BLUE_DARK, HUD_CYAN_LIGHT, HUD_OUTLINE_WIDTH, HUD_CORNER_RADIUS)
        draw_hud_text(draw, "OBJECT CLASSIFICATION", (panel_x_obj + 20, panel_y_obj + 15), font_sub, HUD_TEXT_COLOR_PRIMARY)
        draw_glowing_line(draw, panel_x_obj + 20, panel_y_obj + 50, panel_x_obj + panel_width_obj - 20, panel_y_obj + 50, HUD_CYAN_LIGHT, base_width=1)

        object_counts: Dict[str, int] = {}
        
        # The YOLO model results (already run for logic above) can be reused for drawing bboxes
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
                    
                    draw.rectangle([x1, y1, x2, y2], outline=color_with_alpha, width=max(1, int(frame_width * 0.0015)))
                    
                    text_label = f"{label.upper()} ({conf:.2f})"
                    bbox_text = font_small.getbbox(text_label)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                    
                    text_x = x1 + 4
                    text_y = y1 - text_height - 6
                    if text_y < 0: text_y = y1 + 2 
                        
                    draw_rounded_rectangle(draw, [text_x - 2, text_y - 2, text_x + text_width + 4, text_y + text_height + 4], radius=4, fill=color_with_alpha)
                    draw.text((text_x, text_y), text_label, fill=(0,0,0), font=font_small)

        # Populate object panel content
        obj_content_y_start = panel_y_obj + 60
        obj_line_height = font_small_height + 5
        max_lines_obj = (panel_height_obj - 60) // obj_line_height
        
        current_obj_lines_count = 0
        for obj_label, count in object_counts.items():
            if current_obj_lines_count < max_lines_obj:
                draw_hud_text(draw, f"- {obj_label.capitalize()}: {count}", (panel_x_obj + 20, obj_content_y_start + current_obj_lines_count * obj_line_height), font_small, HUD_TEXT_COLOR_SECONDARY)
                current_obj_lines_count += 1
        
        total_detected_objects = sum(object_counts.values())
        if current_obj_lines_count < max_lines_obj:
             draw_hud_text(draw, f"TOTAL: {total_detected_objects}", (panel_x_obj + 20, obj_content_y_start + current_obj_lines_count * obj_line_height), font_small, HUD_TEXT_COLOR_HIGHLIGHT)


        # --- Scene Confidence Bar Graph (Bottom of Top Right Panel) ---
        graph_y_start = panel_y_obj + panel_height_obj - 100 
        graph_height = 80 
        
        draw_hud_text(draw, "SCENE CONFIDENCE:", (panel_x_obj + 20, graph_y_start - font_small_height - 5), font_small, HUD_TEXT_COLOR_PRIMARY)
        draw_bar_graph(draw, panel_x_obj + 10, graph_y_start, panel_width_obj - 20, graph_height, top_predictions_for_graph, font_small, HUD_CYAN_LIGHT)


        # --- Event Log Panel (Bottom Left) ---
        log_panel_width = 380 
        log_panel_height = 200 
        log_x = 10
        log_y = frame_height - log_panel_height - 20 
        
        draw_hud_box(draw, (log_x, log_y, log_x + log_panel_width, log_y + log_panel_height), HUD_BLUE_DARK, HUD_BLUE_LIGHT, HUD_OUTLINE_WIDTH, HUD_CORNER_RADIUS)
        draw_hud_text(draw, "EVENT LOG", (log_x + 20, log_y + 15), font_sub, HUD_TEXT_COLOR_PRIMARY)
        draw_glowing_line(draw, log_x + 20, log_y + 50, log_x + log_panel_width - 20, log_y + 50, HUD_BLUE_LIGHT, base_width=1)

        log_content_y_start = log_y + 60
        log_line_height = font_small_height + 5 
        max_log_lines = (log_panel_height - 60) // log_line_height

        if len(event_log_history) > 0:
            # Ensure proper scrolling when there are events
            effective_log_length = max_log_lines if len(event_log_history) > max_log_lines else len(event_log_history)
            scroll_duration_frames = effective_log_length * 30 # Slower scroll based on number of lines
            
            # Prevent ZeroDivisionError for modulo if scroll_duration_frames is zero
            if scroll_duration_frames == 0:
                scroll_denominator = 1 # Use a non-zero value, essentially no scroll for very few logs
            else:
                scroll_denominator = scroll_duration_frames

            log_scroll_pos = (frame_counter_for_animation % scroll_denominator) / scroll_duration_frames # Normalized 0.0 to 1.0
            
            # Calculate start index for display, allowing the last lines to scroll in
            start_log_index = max(0, len(event_log_history) - max_log_lines)
            
            # Apply fractional scrolling for a smoother effect
            fractional_offset = (len(event_log_history) - max_log_lines) * log_scroll_pos
            
            for i in range(max_log_lines):
                # Calculate the actual index in the history for this display line
                actual_log_index = start_log_index + i + int(fractional_offset)
                
                if actual_log_index < len(event_log_history):
                    log_entry = event_log_history[actual_log_index]
                    # Adjust y-position by the fractional part for smooth scroll
                    y_pos = log_content_y_start + i * log_line_height - (fractional_offset - int(fractional_offset)) * log_line_height
                    draw_hud_text(draw, log_entry, (log_x + 20, y_pos), font_small, HUD_TEXT_COLOR_SECONDARY)
        else: 
            draw_hud_text(draw, "No events to display.", (log_x + 20, log_content_y_start), font_small, HUD_TEXT_COLOR_SECONDARY)


        # --- System Status Panel (Bottom Right) ---
        sys_panel_width = 300 
        sys_panel_height = 180 
        sys_x = frame_width - sys_panel_width - 20
        sys_y = frame_height - sys_panel_height - 20

        draw_hud_box(draw, (sys_x, sys_y, sys_x + sys_panel_width, sys_y + sys_panel_height), HUD_BLUE_DARK, HUD_GREEN_LIGHT, HUD_OUTLINE_WIDTH, HUD_CORNER_RADIUS)
        draw_hud_text(draw, "SYSTEM HEALTH", (sys_x + 20, sys_y + 15), font_sub, HUD_TEXT_COLOR_PRIMARY)
        draw_glowing_line(draw, sys_x + 20, sys_y + 50, sys_x + sys_panel_width - 20, sys_y + 50, HUD_GREEN_LIGHT, base_width=1)

        fps_text = ""
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            fps_text = f"{fps:.1f}"
        
        simulated_cpu_load = 50 + 20 * math.sin(frame_counter_for_animation * 0.05) 
        simulated_gpu_load = 60 + 15 * math.cos(frame_counter_for_animation * 0.07) 
        simulated_data_rate = 10 + 5 * math.sin(frame_counter_for_animation * 0.03)

        sys_lines = [
            f"Frames Processed: {frame_count}",
            f"Current FPS: {fps_text}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            f"CPU Load: {simulated_cpu_load:.1f}%",
            f"GPU Load: {simulated_gpu_load:.1f}%",
            f"Data Rate: {simulated_data_rate:.1f} MB/s",
            f"Device: {str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).upper()}"
        ]
        
        sys_content_y_start = sys_y + 60
        sys_line_height = font_small_height + 5 
        for i, line in enumerate(sys_lines):
            draw_hud_text(draw, line, (sys_x + 20, sys_content_y_start + i * sys_line_height), font_small, HUD_TEXT_COLOR_SECONDARY)


        # --- Composite the HUD layer onto the original frame ---
        frame_rgba = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        pil_combined_image = Image.alpha_composite(frame_rgba, hud_layer)

        # Convert back to OpenCV format (BGR) for display
        frame_processed_cv2 = cv2.cvtColor(np.array(pil_combined_image), cv2.COLOR_RGBA2BGR)

        # Display the resulting frame
        cv2.imshow("AI-Powered Traffic Monitoring (JARVIS-Level HUD)", frame_processed_cv2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[JARVIS-LOG] Traffic monitoring terminated. All systems offline.")


if __name__ == "__main__":
    run_traffic_monitoring()
