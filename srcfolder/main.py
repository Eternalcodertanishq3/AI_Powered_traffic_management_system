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
# Significantly lowered to capture more objects, relying on tracking/smoothing to filter
YOLO_DETECTION_CONFIDENCE_OVERRIDE = 0.15 # Was 0.25

# --- JARVIS FEATURES CONFIG ---
# Feature 1: Persistent Object Tracking
IOU_THRESHOLD_FOR_TRACKING = 0.3 # IoU threshold to consider a new detection as the same tracked object
MAX_TRACKING_AGE = 10 # How many frames to keep a track without new detection before deleting

# Feature 2: Detection Confidence Smoothing
CONFIDENCE_SMOOTHING_WINDOW_SIZE = 5 # Number of frames to average confidence over for each tracked object


# --- UI Styling (Iron Man / JARVIS Vibe) ---
# RGBA colors for transparency (even more transparent for non-critical states)
HUD_BLUE_DARK_TRANSPARENT = (10, 20, 50, 150) 
HUD_BLUE_MEDIUM_TRANSPARENT = (20, 60, 100, 120) 
HUD_BLUE_LIGHT = (30, 144, 255, 180)    
HUD_CYAN_LIGHT = (0, 255, 255, 180)     
HUD_GREEN_LIGHT = (0, 255, 127, 180)    
HUD_YELLOW_ACCENT = (255, 255, 0, 180)  
HUD_RED_CRITICAL = (255, 69, 0, 220)    
HUD_TEXT_COLOR_PRIMARY = (220, 240, 255, 255) 
HUD_TEXT_COLOR_SECONDARY = (180, 200, 255, 255) 
HUD_TEXT_COLOR_HIGHLIGHT = (0, 255, 255, 255) 
HUD_OUTLINE_WIDTH_BASE = 2 
HUD_CORNER_RADIUS_BASE = 15 


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

# --- Object Tracking Globals ---
# Each tracked_object will be a dict:
# { 'id': unique_id, 'bbox': [x1, y1, x2, y2], 'label': 'car', 'last_seen': frame_num, 'confidence_history': deque }
tracked_objects = {}
next_object_id = 0

# --- Helper function for IoU (Intersection over Union) ---
def calculate_iou(box1, box2):
    """Calculates IoU of two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area

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

def draw_wireframe_element(draw: ImageDraw.ImageDraw, frame_width: int, frame_height: int, line_color: Tuple[int, int, int, int], base_thickness: int, animation_frame: int, ui_scale_factor: float):
    """Draws abstract wireframe elements with subtle animation, adapted for UI scale."""
    scaled_thickness = max(1, int(base_thickness * ui_scale_factor))

    corner_line_length_h = int(frame_width * 0.04) 
    corner_line_length_v = int(frame_height * 0.06) 

    coords = [
        (0, corner_line_length_v, corner_line_length_h, 0), 
        (0, corner_line_length_v * 2, corner_line_length_h * 2, 0),
        (frame_width, corner_line_length_v, frame_width - corner_line_length_h, 0), 
        (frame_width, corner_line_length_v * 2, frame_width - corner_line_length_h * 2, 0),
        (0, frame_height - corner_line_length_v, corner_line_length_h, frame_height), 
        (0, frame_height - corner_line_length_v * 2, corner_line_length_h * 2, frame_height),
        (frame_width, frame_height - corner_line_length_v, frame_width - corner_line_length_h, frame_height), 
        (frame_width, frame_height - corner_line_length_v * 2, frame_width - corner_line_length_h * 2, frame_height)
    ]
    for x1, y1, x2, y2 in coords:
        draw_glowing_line(draw, x1, y1, x2, y2, line_color, scaled_thickness)

    grid_alpha = int(line_color[3] * 0.2) 
    grid_color = line_color[:3] + (grid_alpha,)
    
    num_h_lines = max(3, int(frame_height / (150 * ui_scale_factor))) 
    for i in range(num_h_lines):
        y_offset_base = (animation_frame % 200) * (50.0 / 200.0) 
        y_pos = int((i * (frame_height / num_h_lines)) + y_offset_base) % frame_height
        draw_glowing_line(draw, 0, y_pos, frame_width, y_pos, grid_color, max(1, scaled_thickness // 2))

    num_v_lines = max(3, int(frame_width / (150 * ui_scale_factor))) 
    for i in range(num_v_lines):
        x_offset_base = (animation_frame % 200) * (50.0 / 200.0)
        x_pos = int((i * (frame_width / num_v_lines)) + x_offset_base) % frame_width
        draw_glowing_line(draw, x_pos, 0, x_pos, frame_height, grid_color, max(1, scaled_thickness // 2))

    pulse_radius_max = min(frame_width, frame_height) // 4
    pulse_radius = int(pulse_radius_max * (math.sin(animation_frame * 0.05) * 0.5 + 0.5)) 
    pulse_alpha = int(line_color[3] * (1 - (pulse_radius / pulse_radius_max if pulse_radius_max > 0 else 0)) * 0.7) 
    pulse_color = line_color[:3] + (pulse_alpha,)
    draw.ellipse([frame_width//2 - pulse_radius, frame_height//2 - pulse_radius, 
                  frame_width//2 + pulse_radius, frame_height//2 + pulse_radius], 
                 outline=pulse_color, width=max(1, scaled_thickness))
    
def draw_bar_graph(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, values: List[Dict[str, Any]], font: ImageFont.FreeTypeFont, base_color: Tuple[int,int,int,int], ui_scale_factor: float):
    """Draws a simple bar graph for confidences, adapted for UI scale."""
    bar_spacing = max(1, int(width * 0.01)) 
    
    font_bbox = font.getbbox("Tg")
    font_height = font_bbox[3] - font_bbox[1]

    max_bar_height = height - (font_height * 2) - (bar_spacing * 2)
    max_bar_height = max(1, max_bar_height) 

    num_bars = len(values)
    if num_bars == 0: return

    individual_bar_width = (width - (num_bars + 1) * bar_spacing) // num_bars
    individual_bar_width = max(1, individual_bar_width) 

    for i, pred_dict in enumerate(values):
        label = pred_dict["label"]
        value = pred_dict["confidence"]

        bar_height_actual = int(max_bar_height * value)
        bar_x1 = x + bar_spacing + i * (individual_bar_width + bar_spacing)
        bar_y1 = y + height - bar_height_actual - font_height - bar_spacing
        bar_x2 = bar_x1 + individual_bar_width
        bar_y2 = y + height - font_height - bar_spacing

        bar_color = SCENE_LABEL_COLORS.get(label, base_color)
        draw_rounded_rectangle(draw, (bar_x1, bar_y1, bar_x2, bar_y2), max(1, int(3 * ui_scale_factor)), fill=bar_color+(150,), outline=bar_color, width=1)
        
        draw_hud_text(draw, f"{value:.2f}", (bar_x1, bar_y1 - font_height - max(1, int(2 * ui_scale_factor))), font, HUD_TEXT_COLOR_PRIMARY)
        
        display_label = label.replace('_', ' ')
        label_bbox = font.getbbox(display_label)
        text_w = label_bbox[2] - label_bbox[0]
        
        if text_w > individual_bar_width:
            avg_char_width = text_w / len(display_label) if len(display_label) > 0 else 1
            chars_to_fit = int(individual_bar_width / avg_char_width) - 1 
            if chars_to_fit > 0:
                display_label = display_label[:max(chars_to_fit, 1)].strip() + "." 
            else:
                display_label = "" 
        
        draw_hud_text(draw, display_label, (bar_x1, bar_y2 + max(1, int(2 * ui_scale_factor))), font, HUD_TEXT_COLOR_SECONDARY)


def run_traffic_monitoring():
    global current_alert_level, consecutive_accident_frames, last_alert_timestamp, frame_counter_for_animation
    global tracked_objects, next_object_id # Access global tracking variables

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
    
    orig_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    TARGET_MAX_DISPLAY_WIDTH = 1280 
    TARGET_MAX_DISPLAY_HEIGHT = 720 
    TARGET_MIN_DISPLAY_WIDTH = 800
    TARGET_MIN_DISPLAY_HEIGHT = 600

    aspect_ratio = orig_frame_width / orig_frame_height

    display_width = orig_frame_width
    display_height = orig_frame_height

    if display_width > TARGET_MAX_DISPLAY_WIDTH or display_height > TARGET_MAX_DISPLAY_HEIGHT:
        scale_factor = min(TARGET_MAX_DISPLAY_WIDTH / display_width, TARGET_MAX_DISPLAY_HEIGHT / display_height)
        display_width = int(display_width * scale_factor)
        display_height = int(display_height * scale_factor)
    
    if display_width < TARGET_MIN_DISPLAY_WIDTH or display_height < TARGET_MIN_DISPLAY_HEIGHT:
        scale_factor = max(TARGET_MIN_DISPLAY_WIDTH / display_width, TARGET_MIN_DISPLAY_HEIGHT / display_height)
        display_width = int(display_width * scale_factor)
        display_height = int(display_height * scale_factor)

    final_aspect_ratio = display_width / display_height
    if abs(final_aspect_ratio - aspect_ratio) > 0.01: 
        if final_aspect_ratio > aspect_ratio: 
            display_width = int(display_height * aspect_ratio)
        else: 
            display_height = int(display_width / aspect_ratio)

    frame_width = int(display_width)
    frame_height = int(display_height)

    UI_DESIGN_BASE_WIDTH = 1920 
    UI_DESIGN_BASE_HEIGHT = 1080 

    global_ui_scale_w = frame_width / UI_DESIGN_BASE_WIDTH
    global_ui_scale_h = frame_height / UI_DESIGN_BASE_HEIGHT
    global_ui_scale = min(global_ui_scale_w, global_ui_scale_h) 

    hud_font_size_main_scaled = max(14, int(28 * global_ui_scale)) 
    hud_font_size_sub_scaled = max(10, int(20 * global_ui_scale)) 
    hud_font_size_small_scaled = max(8, int(14 * global_ui_scale)) 

    font_main = ImageFont.truetype(default_font_path, hud_font_size_main_scaled) if os.path.exists(default_font_path) else ImageFont.load_default()
    font_sub = ImageFont.truetype(default_font_path, hud_font_size_sub_scaled) if os.path.exists(default_font_path) else ImageFont.load_default()
    font_small = ImageFont.truetype(default_font_path, hud_font_size_small_scaled) if os.path.exists(default_font_path) else ImageFont.load_default()

    font_main_height = font_main.getbbox("Tg")[3] - font_main.getbbox("Tg")[1]
    font_sub_height = font_sub.getbbox("Tg")[3] - font_sub.getbbox("Tg")[1]
    font_small_height = font_small.getbbox("Tg")[3] - font_small.getbbox("Tg")[1]

    cv2.namedWindow("AI-Powered Traffic Monitoring (JARVIS-Level HUD)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI-Powered Traffic Monitoring (JARVIS-Level HUD)", frame_width, frame_height)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("[JARVIS-LOG] End of video or stream disconnected. Initiating shutdown sequence.")
            break

        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

        frame_count += 1
        frame_counter_for_animation += 1

        if frame_count % FRAME_PROCESS_INTERVAL != 0:
            continue 

        hud_layer = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(hud_layer)

        draw_wireframe_element(draw, frame_width, frame_height, HUD_BLUE_LIGHT, base_thickness=2, animation_frame=frame_counter_for_animation, ui_scale_factor=global_ui_scale)


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
        
        # --- Object Detection & Tracking (JARVIS Feature 1 & 2) ---
        yolo_results_current_frame = yolo_model(pil_frame_rgb, conf=YOLO_DETECTION_CONFIDENCE_OVERRIDE, verbose=False)
        
        current_frame_detections = []
        for r in yolo_results_current_frame:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                
                if label in YOLO_CLASSES: # Only consider relevant classes
                    current_frame_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': label,
                        'confidence': conf
                    })

        # Update tracked objects
        matched_track_ids = set()
        new_detections_to_add = []

        for det in current_frame_detections:
            best_iou = 0.0
            best_match_id = -1
            
            # Find best match in existing tracked objects
            for track_id, track_obj in tracked_objects.items():
                iou = calculate_iou(track_obj['bbox'], det['bbox'])
                if iou > best_iou and iou > IOU_THRESHOLD_FOR_TRACKING and track_obj['label'] == det['label']:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id != -1:
                # Update existing track
                tracked_objects[best_match_id]['bbox'] = det['bbox'] # Update bbox to current detection
                tracked_objects[best_match_id]['last_seen'] = frame_count
                tracked_objects[best_match_id]['confidence_history'].append(det['confidence'])
                matched_track_ids.add(best_match_id)
            else:
                # New detection, add to list for new tracks
                new_detections_to_add.append(det)
        
        # Add new tracks for unmatched detections
        for det in new_detections_to_add:
            tracked_objects[next_object_id] = {
                'id': next_object_id,
                'bbox': det['bbox'],
                'label': det['label'],
                'last_seen': frame_count,
                'confidence_history': deque([det['confidence']], maxlen=CONFIDENCE_SMOOTHING_WINDOW_SIZE)
            }
            next_object_id += 1
        
        # Remove old tracks
        tracks_to_delete = [track_id for track_id, track_obj in tracked_objects.items() if (frame_count - track_obj['last_seen']) > MAX_TRACKING_AGE]
        for track_id in tracks_to_delete:
            del tracked_objects[track_id]

        # Prepare objects for display and alert logic
        display_objects = []
        for track_id, track_obj in tracked_objects.items():
            smoothed_conf = sum(track_obj['confidence_history']) / len(track_obj['confidence_history'])
            
            # Only display objects with smoothed confidence above a reasonable threshold
            # This is the final filter for display and counting
            if smoothed_conf >= YOLO_DETECTION_CONFIDENCE: # Use the original YOLO_DETECTION_CONFIDENCE from constants for final display
                display_objects.append({
                    'id': track_obj['id'],
                    'bbox': track_obj['bbox'],
                    'label': track_obj['label'],
                    'confidence': smoothed_conf # Use smoothed confidence for display
                })
                
                # Update counts for alert logic
                if track_obj['label'] in ["car", "truck", "bus"]: 
                    current_cars += 1
                if track_obj['label'] in ACCIDENT_IMPACT_OBJECTS: # No confidence check here, already filtered by smoothed_conf
                    current_accident_impact_objects += 1

        # Alert Logic (remains largely the same, but uses current_cars/impact_objects from tracked data)
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
        min_display_dim = min(frame_width, frame_height)
        dynamic_padding = max(5, int(20 * (min_display_dim / UI_DESIGN_BASE_HEIGHT))) 
        
        hud_outline_width = max(1, int(HUD_OUTLINE_WIDTH_BASE * global_ui_scale))
        hud_corner_radius = max(5, int(HUD_CORNER_RADIUS_BASE * global_ui_scale))


        panel_base_width_ratio = 0.25 
        panel_base_height_ratio = 0.25 

        # --- Top-left HUD block: Scene, Status, and Action ---
        panel1_width = max(int(frame_width * panel_base_width_ratio), int(160 * global_ui_scale)) 
        panel1_height = max(int(frame_height * panel_base_height_ratio), int(130 * global_ui_scale)) 
        panel1_x = dynamic_padding
        panel1_y = dynamic_padding

        draw_hud_box(draw, (panel1_x, panel1_y, panel1_x + panel1_width, panel1_y + panel1_height), HUD_BLUE_DARK_TRANSPARENT, HUD_BLUE_LIGHT, hud_outline_width, hud_corner_radius)
        
        current_y_in_panel1 = panel1_y + int(10 * global_ui_scale) 

        scene_color_for_display = SCENE_LABEL_COLORS.get(most_common_scene, HUD_TEXT_COLOR_PRIMARY)
        scene_display_text = f"SCENE: {most_common_scene.replace('_', ' ').upper()}"
        
        pulse_alpha = int(255 * (math.sin(frame_counter_for_animation * 0.1) * 0.2 + 0.8)) 
        text_color_pulsating = scene_color_for_display[:3] + (pulse_alpha,)

        draw_hud_text(draw, scene_display_text, (panel1_x + int(20 * global_ui_scale), current_y_in_panel1), font_main, text_color_pulsating)
        current_y_in_panel1 += font_main_height + int(5 * global_ui_scale)
        draw_hud_text(draw, f"CONF: {smoothed_scene_confidence:.2f}", (panel1_x + int(20 * global_ui_scale), current_y_in_panel1), font_sub, HUD_TEXT_COLOR_HIGHLIGHT)
        
        line_start_x = panel1_x + int(15 * global_ui_scale)
        line_end_x = panel1_x + panel1_width - int(15 * global_ui_scale)
        line_y = current_y_in_panel1 + font_sub_height + int(15 * global_ui_scale)
        draw_glowing_line(draw, line_start_x, line_y, line_end_x, line_y, HUD_CYAN_LIGHT, base_width=max(1, int(2 * global_ui_scale)))
        
        scan_x = line_start_x + int((line_end_x - line_start_x) * (frame_counter_for_animation % 60 / 60.0))
        draw_glowing_line(draw, scan_x, line_y - int(5 * global_ui_scale), scan_x, line_y + int(5 * global_ui_scale), HUD_CYAN_LIGHT, base_width=max(1, int(1 * global_ui_scale)))

        current_y_in_panel1 = line_y + int(10 * global_ui_scale)

        alert_text_color = HUD_TEXT_COLOR_PRIMARY
        alert_bg_color = HUD_BLUE_DARK_TRANSPARENT
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


        # --- Object Detection Panel (Top Right) ---
        panel2_width = max(int(frame_width * panel_base_width_ratio), int(180 * global_ui_scale)) 
        panel2_height = max(int(frame_height * 0.35), int(220 * global_ui_scale)) 
        panel2_x = frame_width - panel2_width - dynamic_padding
        panel2_y = dynamic_padding

        draw_hud_box(draw, (panel2_x, panel2_y, panel2_x + panel2_width, panel2_y + panel2_height), HUD_BLUE_DARK_TRANSPARENT, HUD_CYAN_LIGHT, hud_outline_width, hud_corner_radius)
        
        title_text_obj = "OBJECT CLASSIFICATION"
        font_sub_obj_title = font_sub 
        title_text_obj_bbox = font_sub_obj_title.getbbox(title_text_obj)
        text_w = title_text_obj_bbox[2] - title_text_obj_bbox[0] 
        if text_w > (panel2_width - int(40 * global_ui_scale)):
             font_sub_obj_title = ImageFont.truetype(default_font_path, max(8, int(hud_font_size_sub_scaled * 0.8 * (panel2_width / (text_w if text_w > 0 else 1.0)))))
        
        draw_hud_text(draw, title_text_obj, (panel2_x + int(20 * global_ui_scale), panel2_y + int(15 * global_ui_scale)), font_sub_obj_title, HUD_TEXT_COLOR_PRIMARY)
        draw_glowing_line(draw, panel2_x + int(20 * global_ui_scale), panel2_y + int(50 * global_ui_scale), panel2_x + panel2_width - int(20 * global_ui_scale), panel2_y + int(50 * global_ui_scale), HUD_CYAN_LIGHT, base_width=max(1, int(1 * global_ui_scale)))

        # Populate object panel content from tracked_objects
        # Filter for display confidence and count for panel
        object_counts_display: Dict[str, int] = {}
        for obj in display_objects: # Use display_objects which are already filtered and smoothed
            object_counts_display[obj['label']] = object_counts_display.get(obj['label'], 0) + 1

        obj_content_y_start = panel2_y + int(60 * global_ui_scale)
        obj_line_height = font_small_height + int(5 * global_ui_scale)
        
        graph_title_bbox = font_small.getbbox("SCENE CONFIDENCE:") 
        graph_title_height_actual = graph_title_bbox[3] - graph_title_bbox[1] 
        graph_height_actual = int(panel2_height * 0.3) 
        
        available_height_for_obj_list = panel2_height - (obj_content_y_start - panel2_y) - graph_title_height_actual - graph_height_actual - int(10 * global_ui_scale)
        max_lines_obj = max(1, available_height_for_obj_list // obj_line_height)

        current_obj_lines_count = 0
        sorted_objects_display = sorted(object_counts_display.items(), key=lambda item: item[1], reverse=True) 
        for obj_label, count in sorted_objects_display:
            if current_obj_lines_count < max_lines_obj:
                display_obj_text = f"- {obj_label.capitalize()}: {count}"
                obj_text_bbox = font_small.getbbox(display_obj_text) 
                text_w = obj_text_bbox[2] - obj_text_bbox[0] 
                if text_w > (panel2_width - int(40 * global_ui_scale)): 
                    chars_to_fit = int(len(display_obj_text) * ((panel2_width - int(40 * global_ui_scale)) / (text_w if text_w > 0 else 1.0))) - 2
                    if chars_to_fit > 0:
                        display_obj_text = display_obj_text[:max(chars_to_fit, 1)].strip() + "."
                    else:
                        display_obj_text = "" 
                draw_hud_text(draw, display_obj_text, (panel2_x + int(20 * global_ui_scale), obj_content_y_start + current_obj_lines_count * obj_line_height), font_small, HUD_TEXT_COLOR_SECONDARY)
                current_obj_lines_count += 1
        
        total_detected_objects = sum(object_counts_display.values()) # Use display_objects for total count
        if current_obj_lines_count < max_lines_obj and max_lines_obj > 0: 
             draw_hud_text(draw, f"TOTAL: {total_detected_objects}", (panel2_x + int(20 * global_ui_scale), obj_content_y_start + current_obj_lines_count * obj_line_height), font_small, HUD_TEXT_COLOR_HIGHLIGHT)


        # --- Scene Confidence Bar Graph (Bottom of Top Right Panel) ---
        graph_y_start = panel2_y + panel2_height - graph_height_actual - int(10 * global_ui_scale)
        
        draw_hud_text(draw, "SCENE CONFIDENCE:", (panel2_x + int(20 * global_ui_scale), graph_y_start - graph_title_height_actual), font_small, HUD_TEXT_COLOR_PRIMARY)
        draw_bar_graph(draw, panel2_x + int(10 * global_ui_scale), graph_y_start, panel2_width - int(20 * global_ui_scale), graph_height_actual, top_predictions_for_graph, font_small, HUD_CYAN_LIGHT, ui_scale_factor=global_ui_scale)


        # --- Event Log Panel (Bottom Left) ---
        panel3_width = max(int(frame_width * panel_base_width_ratio), int(160 * global_ui_scale)) 
        panel3_height = max(int(frame_height * panel_base_height_ratio), int(130 * global_ui_scale))
        panel3_x = dynamic_padding
        panel3_y = frame_height - panel3_height - dynamic_padding
        
        draw_hud_box(draw, (panel3_x, panel3_y, panel3_x + panel3_width, panel3_y + panel3_height), HUD_BLUE_DARK_TRANSPARENT, HUD_BLUE_LIGHT, hud_outline_width, hud_corner_radius)
        draw_hud_text(draw, "EVENT LOG", (panel3_x + int(20 * global_ui_scale), panel3_y + int(15 * global_ui_scale)), font_sub, HUD_TEXT_COLOR_PRIMARY)
        draw_glowing_line(draw, panel3_x + int(20 * global_ui_scale), panel3_y + int(50 * global_ui_scale), panel3_x + panel3_width - int(20 * global_ui_scale), panel3_y + int(50 * global_ui_scale), HUD_BLUE_LIGHT, base_width=max(1, int(1 * global_ui_scale)))

        log_content_y_start = panel3_y + int(60 * global_ui_scale)
        log_line_height = font_small_height + int(5 * global_ui_scale) 
        available_height_for_log = panel3_height - (log_content_y_start - panel3_y) - int(10 * global_ui_scale)
        max_log_lines = max(1, available_height_for_log // log_line_height)

        if len(event_log_history) > 0:
            effective_log_length = len(event_log_history)
            scroll_speed_factor = max(1.0, effective_log_length / max_log_lines) if max_log_lines > 0 else 1.0
            scroll_duration_frames = int(effective_log_length * 30 / scroll_speed_factor) 
            
            if scroll_duration_frames == 0: scroll_denominator = 1 
            else: scroll_denominator = scroll_duration_frames

            log_scroll_pos_norm = (frame_counter_for_animation % scroll_denominator) / scroll_denominator
            
            total_scroll_lines = (effective_log_length - max_log_lines) if effective_log_length > max_log_lines else 0
            current_scroll_offset_lines = total_scroll_lines * log_scroll_pos_norm
            
            for i in range(max_log_lines):
                log_entry_target_index = i + current_scroll_offset_lines
                actual_log_index = int(log_entry_target_index)
                
                if actual_log_index < effective_log_length and actual_log_index >= 0:
                    log_entry = event_log_history[actual_log_index]
                    
                    y_pos_adjustment = (log_entry_target_index - actual_log_index) * log_line_height
                    y_pos = log_content_y_start + i * log_line_height - y_pos_adjustment
                    
                    display_log_text = log_entry
                    log_text_bbox = font_small.getbbox(display_log_text) 
                    text_w = log_text_bbox[2] - log_text_bbox[0] 
                    if text_w > (panel3_width - int(40 * global_ui_scale)):
                        chars_to_fit = int(len(display_log_text) * ((panel3_width - int(40 * global_ui_scale)) / (text_w if text_w > 0 else 1.0))) - 2
                        if chars_to_fit > 0:
                            display_log_text = display_log_text[:max(chars_to_fit, 1)].strip() + "."
                        else:
                            display_log_text = ""
                    draw_hud_text(draw, display_log_text, (panel3_x + int(20 * global_ui_scale), y_pos), font_small, HUD_TEXT_COLOR_SECONDARY)
        else: 
            draw_hud_text(draw, "No events to display.", (panel3_x + int(20 * global_ui_scale), log_content_y_start), font_small, HUD_TEXT_COLOR_SECONDARY)


        # --- System Status Panel (Bottom Right) ---
        panel4_width = max(int(frame_width * 0.25), int(160 * global_ui_scale)) 
        panel4_height = max(int(frame_height * panel_base_height_ratio), int(130 * global_ui_scale)) 
        panel4_x = frame_width - panel4_width - dynamic_padding
        panel4_y = frame_height - panel4_height - dynamic_padding

        draw_hud_box(draw, (panel4_x, panel4_y, panel4_x + panel4_width, panel4_y + panel4_height), HUD_BLUE_DARK_TRANSPARENT, HUD_GREEN_LIGHT, hud_outline_width, hud_corner_radius)
        
        title_text_sys = "SYSTEM HEALTH"
        font_sub_sys_title = font_sub
        sys_title_bbox = font_sub_sys_title.getbbox(title_text_sys) 
        text_w = sys_title_bbox[2] - sys_title_bbox[0] 
        if text_w > (panel4_width - int(40 * global_ui_scale)):
            font_sub_sys_title = ImageFont.truetype(default_font_path, max(8, int(hud_font_size_sub_scaled * 0.8 * (panel4_width / (text_w if text_w > 0 else 1.0)))))

        draw_hud_text(draw, title_text_sys, (panel4_x + int(20 * global_ui_scale), panel4_y + int(15 * global_ui_scale)), font_sub_sys_title, HUD_TEXT_COLOR_PRIMARY)
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
        
        available_height_for_sys_list = panel4_height - (sys_content_y_start - panel4_y) - int(10 * global_ui_scale)
        max_sys_lines = max(1, available_height_for_sys_list // sys_line_height)

        for i, line in enumerate(sys_lines):
            if i < max_sys_lines: 
                display_sys_text = line
                sys_text_bbox = font_small.getbbox(display_sys_text) 
                text_w = sys_text_bbox[2] - sys_text_bbox[0] 
                if text_w > (panel4_width - int(40 * global_ui_scale)):
                    chars_to_fit = int(len(display_sys_text) * ((panel4_width - int(40 * global_ui_scale)) / (text_w if text_w > 0 else 1.0))) - 2
                    if chars_to_fit > 0:
                        display_sys_text = display_sys_text[:max(chars_to_fit, 1)].strip() + "."
                    else:
                        display_sys_text = ""
                draw_hud_text(draw, display_sys_text, (panel4_x + int(20 * global_ui_scale), sys_content_y_start + i * sys_line_height), font_small, HUD_TEXT_COLOR_SECONDARY)

        # --- Draw tracked objects (bounding boxes and labels) ---
        for obj in display_objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            label = obj['label']
            conf = obj['confidence']
            obj_id = obj['id'] # Use the tracked object ID

            color = BBOX_COLORS.get(label, (200, 200, 200))
            color_with_alpha = color + (180,) # Slightly more opaque for BBoxes
            
            draw.rectangle([x1, y1, x2, y2], outline=color_with_alpha, width=max(1, int(frame_width * 0.002 * global_ui_scale))) 
            
            text_label = f"{label.upper()} ({conf:.2f}) [ID:{obj_id}]" # Include ID in label
            bbox_font_small = font_small
            
            bbox_label_metrics = bbox_font_small.getbbox(text_label)
            text_w = bbox_label_metrics[2] - bbox_label_metrics[0]
            text_h = bbox_label_metrics[3] - bbox_label_metrics[1]
            
            max_bbox_label_width = x2 - x1 
            if max_bbox_label_width < int(50 * global_ui_scale): 
                max_bbox_label_width = int(50 * global_ui_scale) 
            
            if text_w > max_bbox_label_width:
                chars_to_fit = int(len(text_label) * (max_bbox_label_width / (text_w if text_w > 0 else 1.0))) - 2 
                if chars_to_fit > 0:
                    text_label = text_label[:max(chars_to_fit, 1)].strip()
                    if len(text_label) > 1:
                        text_label += "." 
                else:
                    text_label = "" 


            text_x = x1 + max(1, int(4 * global_ui_scale))
            text_y = y1 - text_h - max(1, int(6 * global_ui_scale))
            if text_y < 0: text_y = y1 + max(1, int(2 * global_ui_scale)) 
                
            draw_rounded_rectangle(draw, [text_x - max(1, int(2 * global_ui_scale)), text_y - max(1, int(2 * global_ui_scale)), text_x + text_w + max(1, int(4 * global_ui_scale)), text_y + text_h + max(1, int(4 * global_ui_scale))], radius=max(1, int(4 * global_ui_scale)), fill=color_with_alpha)
            draw.text((text_x, text_y), text_label, fill=(0,0,0), font=bbox_font_small)


        # --- Composite the HUD layer onto the original frame ---
        frame_rgba = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        pil_combined_image = Image.alpha_composite(frame_rgba, hud_layer)

        cv2.imshow("AI-Powered Traffic Monitoring (JARVIS-Level HUD)", cv2.cvtColor(np.array(pil_combined_image), cv2.COLOR_RGBA2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[JARVIS-LOG] Traffic monitoring terminated. All systems offline.")


if __name__ == "__main__":
    run_traffic_monitoring()
