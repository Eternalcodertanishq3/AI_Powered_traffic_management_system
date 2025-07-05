# srcfolder/constants.py

import os

# --- PATHS ---
# Base directory for the project (assuming this file is in srcfolder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# YOLO Model Path
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

# Scene Classifier Model Path
SCENE_CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "scene_classifier.pth")

# Font Path (Update this if your font is elsewhere or you prefer a different one)
# Ensure this font exists on your system for best results.
# For Windows: "C:/Windows/Fonts/arial.ttf" or "C:/Windows/Fonts/segoeui.ttf"
# For macOS: "/Library/Fonts/Arial.ttf" or "/System/Library/Fonts/Supplemental/Arial.ttf"
# For Linux: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEFAULT_FONT_PATH = "C:/Windows/Fonts/arial.ttf"


# --- VIDEO CONFIGURATION ---
# Frame processing interval (process every Nth frame for performance)
# Higher value means smoother video but less frequent detection updates.
FRAME_PROCESS_INTERVAL = 5 

# Target display resolution for the output window
TARGET_MAX_DISPLAY_WIDTH = 1280 
TARGET_MAX_DISPLAY_HEIGHT = 720 
TARGET_MIN_DISPLAY_WIDTH = 800
TARGET_MIN_DISPLAY_HEIGHT = 600


# --- YOLO DETECTION CONFIG ---
# This is the confidence threshold YOLO uses for its raw detections.
YOLO_DETECTION_CONFIDENCE = 0.25 

# Classes that YOLO is trained to detect and we are interested in.
YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Bounding box colors for different classes (RGB tuples)
BBOX_COLORS = {
    "person": (255, 0, 0),        # Red
    "bicycle": (0, 255, 0),       # Green
    "car": (0, 165, 255),         # Orange
    "motorcycle": (255, 0, 255),  # Magenta
    "bus": (255, 255, 0),         # Yellow
    "truck": (0, 255, 255),       # Cyan
    "traffic light": (255, 128, 0), # Orange-Red
    "stop sign": (0, 0, 255),     # Blue
    "fire": (255, 69, 0),         # Red-Orange (for accident impact)
    "stalled_vehicle": (255, 140, 0), # Dark Orange
    "debris_on_road": (100, 100, 100), # Grey
    "overturned_vehicle": (139, 0, 0), # Dark Red
    "emergency_vehicle_passing": (255, 215, 0), # Gold
    # Add more as needed
}


# --- SCENE CLASSIFICATION CONFIG ---
SCENE_CLASSES = [
    "normal_traffic", "dense_traffic", "sparse_traffic", "accident",
    "emergency_vehicle_passing", "road_block", "stalled_vehicle",
    "pedestrian_crossing", "fire", "adverse_weather_rain",
    "adverse_weather_fog", "adverse_weather_snow", "animal_on_road",
    "debris_on_road", "overturned_vehicle", "multi_vehicle_collision",
    "single_vehicle_incident", "hazard_liquid_spill", "high_speed_chase",
    "traffic_light_malfunction", "road_construction", "post_accident_clearance",
    "damaged_infrastructure"
]

# Scene smoothing window size (number of frames to average for scene prediction)
SCENE_SMOOTHING_WINDOW_SIZE = 20
MIN_SCENE_CONFIDENCE_DISPLAY = 0.50 # Minimum confidence to display a scene prediction

# Accident alert thresholds and persistence
ACCIDENT_CONFIDENCE_THRESHOLD_OBSERVE = 0.70 
ACCIDENT_CONFIDENCE_THRESHOLD_WARN = 0.90 
ACCIDENT_CONFIDENCE_THRESHOLD_CRITICAL_ALERT = 0.95 
ACCIDENT_PERSISTENCE_FRAMES_WARN = 10 
ACCIDENT_PERSISTENCE_FRAMES_CRITICAL = 15 
ALERT_COOLDOWN_SECONDS = 30 

# Smart accident filter
DENSE_TRAFFIC_CAR_COUNT_THRESHOLD_FOR_FALSE_POSITIVE = 15 
ACCIDENT_IMPACT_OBJECTS = ["stalled_vehicle", "debris_on_road", "overturned_vehicle", "fire", "truck", "bus"]
MIN_IMPACT_OBJECTS_FOR_ACCIDENT = 1 


# --- OBJECT TRACKING & ANALYSIS CONFIG ---
IOU_THRESHOLD_FOR_TRACKING = 0.2 
MAX_TRACKING_AGE = 15 # Frames an object can be unmatched before being removed

# Confidence Smoothing for individual objects
CONFIDENCE_SMOOTHING_WINDOW_SIZE = 7 
MIN_CONF_FOR_SMOOTHING_HISTORY = 0.05 

# Trajectory Prediction
TRAJECTORY_PREDICTION_LENGTH = 70 # Pixels to extend trajectory line
TRAJECTORY_SMOOTHING_FACTOR = 0.8 # How much new velocity influences smoothed velocity

# Threat Assessment
THREAT_BASE_SCORES = {
    "person": 50, "bicycle": 30, "car": 20, "motorcycle": 30, "bus": 40, "truck": 40,
    "stalled_vehicle": 80, "debris_on_road": 70, "overturned_vehicle": 100, "fire": 100,
    "emergency_vehicle_passing": 60 
}
THREAT_SPEED_MULTIPLIER = 0.5 # Multiplier for speed's impact on threat
THREAT_DISTANCE_INVERSE_MULTIPLIER = 100 # Multiplier for proximity's impact on threat

# Environmental Anomaly Detection
BRIGHTNESS_CHANGE_THRESHOLD = 30 
DENSITY_CHANGE_THRESHOLD = 0.2 
ENVIRONMENTAL_ALERT_COOLDOWN = 10 # Seconds

# Number Plate Recognition (SIMULATED)
NPR_SIMULATION_CHANCE = 0.1 # 10% chance to "recognize" a plate on a car/truck
PLATE_LOG_MAX_SIZE = 50 # Max number of unique plates to store in the log

# Advanced Lane & Road Boundary Detection
LANE_LINE_COLOR = (0, 255, 0, 100) # Green, transparent
LANE_LINE_THICKNESS = 3 # Base thickness

# Predictive Incident Warning
COLLISION_PROXIMITY_THRESHOLD = 50 # Pixels
COLLISION_ANGLE_THRESHOLD = 150 # Degrees (180 means directly towards each other)
COLLISION_ALERT_COOLDOWN = 5 # Seconds

# Traffic Flow Analysis & Anomaly Detection
TRAFFIC_ZONE_COUNT = 3 # Number of vertical zones for flow analysis
TRAFFIC_FLOW_SPEED_THRESHOLD_SLOWDOWN = 0.3 # Percentage drop from avg speed to be considered slowdown (e.g., 0.3 = 30% drop)
TRAFFIC_FLOW_ALERT_COOLDOWN = 15 # Seconds

# Driver Behavior Monitoring (Simulated)
TAILGATING_DISTANCE_THRESHOLD = 80 # Pixels
TAILGATING_SPEED_DIFF_THRESHOLD = 5 # Pixels/frame (how much faster trailing car is)
AGGRESSIVE_LANE_CHANGE_THRESHOLD = 50 # Pixels horizontal movement in short time (e.g., 5 frames)
DRIVER_BEHAVIOR_ALERT_COOLDOWN = 10 # Seconds


# --- UI STYLING (Iron Man / JARVIS Vibe) ---
UI_DESIGN_BASE_WIDTH = 1920 
UI_DESIGN_BASE_HEIGHT = 1080 

# RGBA colors for transparency (R, G, B, Alpha)
HUD_BLUE_DARK_TRANSPARENT = (10, 20, 50, 20) 
HUD_BLUE_MEDIUM_TRANSPARENT = (20, 60, 100, 10) 
HUD_BLUE_LIGHT = (30, 144, 255, 80) 
HUD_CYAN_LIGHT = (0, 255, 255, 80)     
HUD_GREEN_LIGHT = (0, 255, 127, 80)    
HUD_YELLOW_ACCENT = (255, 255, 0, 80)  
HUD_RED_CRITICAL = (255, 69, 0, 150)    
HUD_TEXT_COLOR_PRIMARY = (255, 255, 255, 255) # Pure white for max visibility
HUD_TEXT_COLOR_SECONDARY = (200, 220, 255, 255) # Lighter blue for secondary
HUD_TEXT_COLOR_HIGHLIGHT = (0, 255, 255, 255) # Pure cyan for highlights

# UI element dimensions
HUD_OUTLINE_WIDTH_BASE = 2 
HUD_CORNER_RADIUS_BASE = 15 
HUD_PADDING_BASE = 20 # Base padding for UI elements

# Text outline for crystal clarity
TEXT_OUTLINE_COLOR = (0, 0, 0, 200) # Dark, semi-transparent outline
TEXT_OUTLINE_WIDTH = 3 # Increased width for maximum clarity

# Panel layout ratios (relative to frame dimensions)
PANEL_BASE_WIDTH_RATIO = 0.22 
PANEL_BASE_HEIGHT_RATIO = 0.20 # Adjusted for more compact panels

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
