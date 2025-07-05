# srcfolder/detection_model.py

import torch
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import random
from typing import Dict, Any, List

# Import constants
from .constants import YOLO_MODEL_PATH, YOLO_DETECTION_CONFIDENCE, SCENE_CLASSIFIER_MODEL_PATH, SCENE_CLASSES

# --- Global Models ---
yolo_model = None
scene_classifier_model = None # Placeholder for scene classification model

def load_models():
    """Loads the YOLO and Scene Classifier models."""
    global yolo_model, scene_classifier_model
    try:
        if yolo_model is None:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[INFO] YOLOv8 model '{YOLO_MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Could not load YOLO model from {YOLO_MODEL_PATH}: {e}")
        print("Please ensure 'yolov8n.pt' is in your 'models' folder.")
        yolo_model = None # Ensure it's None if loading fails

    # Load Scene Classifier Model (Placeholder - actual implementation would load a PyTorch model)
    # For now, we'll simulate scene classification. In a real scenario, you'd load:
    # if scene_classifier_model is None:
    #     scene_classifier_model = torch.load(SCENE_CLASSIFIER_MODEL_PATH)
    #     scene_classifier_model.eval() # Set to evaluation mode
    #     print(f"[INFO] Scene classifier model '{SCENE_CLASSIFIER_MODEL_PATH}' loaded successfully.")
    # For this simulation, we don't need to load a physical model.
    print(f"[INFO] Scene classifier model simulation active. No physical model loaded.")


def get_yolo_detections(pil_image: Image.Image) -> List[Dict[str, Any]]:
    """
    Performs YOLO inference on a PIL image and returns a list of detections.
    Each detection includes bbox, label, and confidence.
    """
    if yolo_model is None:
        print("[ERROR] YOLO model not loaded. Cannot perform detection.")
        return []

    results = yolo_model(pil_image, conf=YOLO_DETECTION_CONFIDENCE, verbose=False)
    
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'label': label,
                'confidence': conf
            })
    return detections


def get_scene_prediction(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Simulates scene classification based on detected objects or provides a random scene.
    In a real scenario, this would involve a trained scene classification model.
    """
    # This is a simulation. In a real application, you would pass the image
    # through your trained scene_classifier_model.
    # Example:
    # if scene_classifier_model is None:
    #     load_models() # Try to load if not loaded
    # if scene_classifier_model:
    #     # Preprocess image for the model
    #     # Run inference
    #     # Interpret results to get main_prediction, main_confidence, all_predictions
    #     pass

    # For simulation, we'll use a simple heuristic or random choice
    # based on the current frame's characteristics or just random.
    
    # Simple simulation:
    # You could analyze objects detected by YOLO here to influence scene prediction.
    # For now, let's just pick a plausible scene.
    
    # Example of a slightly more intelligent simulation based on object count
    detections = get_yolo_detections(pil_image) # Get detections to influence scene
    car_count = sum(1 for d in detections if d['label'] == 'car')
    truck_count = sum(1 for d in detections if d['label'] == 'truck')
    person_count = sum(1 for d in detections if d['label'] == 'person')

    main_prediction = "normal_traffic"
    main_confidence = 0.85
    all_predictions = []

    if car_count > 15 or truck_count > 5:
        main_prediction = "dense_traffic"
        main_confidence = min(0.95, 0.85 + (car_count + truck_count) * 0.01)
    elif car_count < 3 and truck_count == 0 and person_count < 2:
        main_prediction = "sparse_traffic"
        main_confidence = min(0.90, 0.75 + (5 - car_count) * 0.02)
    
    # Introduce a small chance of "accident" or "emergency" for dynamic alerts
    if random.random() < 0.02: # 2% chance for a simulated incident
        simulated_incident = random.choice(["accident", "emergency_vehicle_passing", "stalled_vehicle"])
        main_prediction = simulated_incident
        main_confidence = random.uniform(0.75, 0.99)
        if main_prediction == "accident":
            # Ensure some "impact objects" are simulated if accident is predicted
            if not any(d['label'] in ["stalled_vehicle", "debris_on_road", "overturned_vehicle"] for d in detections):
                # This is where you'd ideally add a "stalled_vehicle" or "debris" detection
                # For now, we just let the scene prediction handle it.
                pass
    
    # Populate all_predictions for the bar graph
    # Ensure top prediction is included and add some plausible others
    all_predictions.append({"label": main_prediction, "confidence": main_confidence})
    
    # Add other random plausible scenes with lower confidence
    other_scenes = [s for s in SCENE_CLASSES if s != main_prediction]
    random.shuffle(other_scenes)
    for i in range(min(4, len(other_scenes))): # Show up to 4 other scenes
        all_predictions.append({"label": other_scenes[i], "confidence": random.uniform(0.05, main_confidence * 0.5)})
    
    all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)

    # Determine suggested action based on main prediction
    suggested_action = "Monitoring traffic flow."
    if main_prediction == "dense_traffic":
        suggested_action = "Advise alternate routes."
    elif main_prediction == "accident":
        suggested_action = "Initiate emergency protocols. Dispatch units."
    elif main_prediction == "emergency_vehicle_passing":
        suggested_action = "Clear path for emergency vehicle."
    elif main_prediction == "road_block":
        suggested_action = "Investigate and clear blockage."
    elif main_prediction == "stalled_vehicle":
        suggested_action = "Dispatch roadside assistance."
    elif main_prediction == "adverse_weather_rain":
        suggested_action = "Issue weather advisory. Advise caution."
    
    return {
        "main_prediction": main_prediction,
        "main_confidence": main_confidence,
        "all_predictions": all_predictions,
        "suggested_action": suggested_action
    }

