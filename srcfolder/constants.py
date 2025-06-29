# srcfolder/constants.py

# Define the scene classes for classification
SCENE_CLASSES = [
    "normal_traffic",
    "dense_traffic",
    "sparse_traffic",
    "accident",
    "emergency_vehicle_passing",
    "road_block",
    "stalled_vehicle",
    "pedestrian_crossing",
    "fire",
    "adverse_weather_rain",
    "adverse_weather_fog",
    "adverse_weather_snow",
    "animal_on_road",
    "debris_on_road",
    "overturned_vehicle",
    "multi_vehicle_collision",
    "single_vehicle_incident",
    "hazard_liquid_spill",
    "high_speed_chase",
    "traffic_light_malfunction",
    "road_construction",
    "post_accident_clearance",
    "damaged_infrastructure",
]

# Path to the trained scene classifier model
SCENE_CLASSIFIER_MODEL_PATH = "models/scene_classifier.pth"

# YOLOv8 object detection model path
YOLO_MODEL_PATH = "models/yolov8n.pt" # Default YOLOv8 Nano model

# Confidence threshold for YOLOv8 object detection (adjust as needed in main.py if you want to override)
YOLO_DETECTION_CONFIDENCE = 0.5 # Default global threshold

# Object classes for YOLOv8 (subset of COCO classes relevant to traffic)
# These are the classes your system will specifically look for and display.
YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "stop sign", 
    # Removed less relevant classes for brevity, you can add them back from YOLO's original 80 if needed:
    # "airplane", "boat", "fire hydrant", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    # "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    # "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    # "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    # "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    # "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    # "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    # "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Colors for bounding boxes (RGB tuple) - matched to Iron Man HUD vibe
BBOX_COLORS = {
    "person": (0, 255, 255),      # Cyan
    "car": (0, 255, 0),           # Green
    "truck": (30, 144, 255),      # Dodger Blue
    "bus": (255, 255, 0),         # Yellow
    "motorcycle": (255, 0, 255),  # Magenta
    "bicycle": (0, 128, 255),     # Orange-Blue
    "traffic light": (255, 165, 0), # Orange
    "stop sign": (255, 0, 0),     # Red
}
