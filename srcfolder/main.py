# srcfolder/main.py

import cv2
import os
import torch
import time
from collections import deque
from datetime import datetime
import numpy as np
from PIL import Image
import math

# Import from our new modular structure
from .constants import (
    VIDEO_INPUT_SOURCE, FRAME_PROCESS_INTERVAL, TARGET_MAX_DISPLAY_WIDTH,
    TARGET_MAX_DISPLAY_HEIGHT, TARGET_MIN_DISPLAY_WIDTH, TARGET_MIN_DISPLAY_HEIGHT,
    UI_DESIGN_BASE_WIDTH, UI_DESIGN_BASE_HEIGHT
)
from .detection_model import load_models, get_yolo_detections, get_scene_prediction
from .traffic_analysis import TrafficAnalyzer
from .hud_renderer import HUDRenderer

def run_traffic_monitoring():
    """
    Main function to run the AI-powered traffic monitoring system.
    Orchestrates detection, analysis, and HUD rendering.
    """
    # --- Initialize Video Capture ---
    cap = None
    if VIDEO_INPUT_SOURCE == "WEBCAM":
        cap = cv2.VideoCapture(0)
    elif os.path.exists(VIDEO_INPUT_SOURCE):
        cap = cv2.VideoCapture(VIDEO_INPUT_SOURCE)
    else:
        print(f"[ERROR] Video input source not found: {VIDEO_INPUT_SOURCE}")
        print("Please check VIDEO_INPUT_SOURCE in constants.py or provide a valid path/option.")
        return

    if not cap.isOpened():
        print("[CRITICAL ERROR] Could not open video source.")
        return

    # --- Load AI Models ---
    load_models()

    # --- Determine Display Resolution and UI Scaling ---
    orig_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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

    global_ui_scale_w = frame_width / UI_DESIGN_BASE_WIDTH
    global_ui_scale_h = frame_height / UI_DESIGN_BASE_HEIGHT
    global_ui_scale = min(global_ui_scale_w, global_ui_scale_h) 

    cv2.namedWindow("AI-Powered Traffic Monitoring (JARVIS-Level HUD)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI-Powered Traffic Monitoring (JARVIS-Level HUD)", frame_width, frame_height)

    # --- Initialize Modules ---
    traffic_analyzer = TrafficAnalyzer(frame_width, frame_height, global_ui_scale)
    hud_renderer = HUDRenderer(frame_width, frame_height)

    # --- Global State for Main Loop ---
    frame_count = 0
    start_time = time.time()
    frame_counter_for_animation = 0 # For UI animations
    event_log_history = deque(maxlen=10) # Central event log
    
    # Variables to hold the latest detection and analysis results
    latest_yolo_detections = []
    latest_scene_report = {
        "main_prediction": "UNAVAILABLE",
        "main_confidence": 0.0,
        "all_predictions": [],
        "suggested_action": "System initializing..."
    }
    latest_alert_level = "OBSERVATION"
    latest_display_action_message = "System initializing..."
    latest_most_common_scene = "UNAVAILABLE"
    latest_smoothed_scene_confidence = 0.0
    
    plate_lookup_active = False
    current_plate_lookup_index = 0
    current_plate_details_display = None

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            # If video ends, reset to beginning to loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

        frame_count += 1
        frame_counter_for_animation += 1

        # --- Process Input ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'): # ROI Tag
            roi_x1_abs = int(frame_width * 0.3)
            roi_y1_abs = int(frame_height * 0.3)
            roi_x2_abs = int(frame_width * 0.7)
            roi_y2_abs = int(frame_height * 0.7)

            objects_in_roi = []
            for obj in traffic_analyzer.get_current_tracked_objects(): 
                bbox = obj['bbox']
                if not (bbox[0] > roi_x2_abs or bbox[2] < roi_x1_abs or bbox[1] > roi_y2_abs or bbox[3] < roi_y1_abs):
                    objects_in_roi.append(f"{obj['label']} (ID:{obj['id']})")
            
            if objects_in_roi:
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - ROI Tagged: {', '.join(objects_in_roi)}")
            else:
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - ROI Tagged: No objects detected in ROI.")
        
        elif key == ord('p'): # Toggle Plate Lookup Mode
            plate_lookup_active = not plate_lookup_active
            current_plate_lookup_index = 0
            current_plate_details_display = None
            if plate_lookup_active:
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Plate Lookup Mode: ON")
                if traffic_analyzer.get_all_detected_plates():
                    selected_plate = traffic_analyzer.get_all_detected_plates()[current_plate_lookup_index]
                    current_plate_details_display = traffic_analyzer.get_simulated_plate_details(selected_plate)
                else:
                    event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - No plates to lookup yet.")
            else:
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Plate Lookup Mode: OFF")

        elif plate_lookup_active and (key == ord('w') or key == 82): # Up arrow (82 for Windows)
            if traffic_analyzer.get_all_detected_plates():
                current_plate_lookup_index = (current_plate_lookup_index - 1) % len(traffic_analyzer.get_all_detected_plates())
                selected_plate = traffic_analyzer.get_all_detected_plates()[current_plate_lookup_index]
                current_plate_details_display = traffic_analyzer.get_simulated_plate_details(selected_plate)
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Selected Plate: {selected_plate}")

        elif plate_lookup_active and (key == ord('s') or key == 84): # Down arrow (84 for Windows)
            if traffic_analyzer.get_all_detected_plates():
                current_plate_lookup_index = (current_plate_lookup_index + 1) % len(traffic_analyzer.get_all_detected_plates())
                selected_plate = traffic_analyzer.get_all_detected_plates()[current_plate_lookup_index]
                current_plate_details_display = traffic_analyzer.get_simulated_plate_details(selected_plate)
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Selected Plate: {selected_plate}")
        
        elif plate_lookup_active and key == 27: # ESC key to close pop-up
            plate_lookup_active = False
            current_plate_details_display = None
            event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Plate Lookup Closed.")


        # --- AI Processing (only every FRAME_PROCESS_INTERVAL frames) ---
        if frame_count % FRAME_PROCESS_INTERVAL == 0:
            pil_frame_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 1. Get YOLO Detections
            latest_yolo_detections = get_yolo_detections(pil_frame_rgb)

            # 2. Update Object Tracking & Get Display Objects
            display_objects = traffic_analyzer.update_object_tracking(latest_yolo_detections, frame_count)
            
            # 3. Get Scene Prediction
            latest_scene_report = get_scene_prediction(pil_frame_rgb)
            
            # 4. Analyze Scene and Alerts
            (latest_alert_level, latest_display_action_message, 
             latest_most_common_scene, latest_smoothed_scene_confidence, _) = \
                traffic_analyzer.analyze_scene_and_alerts(latest_scene_report, event_log_history)

            # 5. Analyze Environmental Anomalies
            traffic_analyzer.analyze_environmental_anomalies(frame, event_log_history)

            # 6. Analyze Predictive Incidents
            traffic_analyzer.analyze_predictive_incidents(event_log_history)

            # 7. Analyze Traffic Flow
            traffic_analyzer.analyze_traffic_flow(event_log_history)

            # 8. Analyze Driver Behavior
            traffic_analyzer.analyze_driver_behavior(event_log_history)

        else:
            # If not processing a full frame, use the last known display_objects
            # and update their 'last_seen_unmatched_count' to correctly age them out.
            # This is crucial for smooth UI even if detection is less frequent.
            display_objects = traffic_analyzer.update_object_tracking([], frame_count) # Pass empty detections to just update age
            # Re-fetch display objects based on the updated internal state
            display_objects = traffic_analyzer.get_current_tracked_objects()


        # --- Prepare System Status for HUD ---
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        system_status = {
            "frame_count": frame_count,
            "fps": fps,
            "cpu_load": 50 + 20 * math.sin(frame_counter_for_animation * 0.05), # Simulated
            "gpu_load": 60 + 15 * math.cos(frame_counter_for_animation * 0.07), # Simulated
            "data_rate": 10 + 5 * math.sin(frame_counter_for_animation * 0.03), # Simulated
            "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).upper()
        }

        # Combine scene analysis data for rendering
        scene_analysis_for_render = {
            "main_prediction": latest_scene_report["main_prediction"],
            "main_confidence": latest_scene_report["main_confidence"],
            "all_predictions": latest_scene_report["all_predictions"],
            "suggested_action": latest_display_action_message,
            "current_alert_level": latest_alert_level,
            "most_common_scene": latest_most_common_scene,
            "smoothed_scene_confidence": latest_smoothed_scene_confidence
        }

        # --- Render HUD ---
        hud_layer = hud_renderer.render_hud(
            frame, 
            display_objects, 
            scene_analysis_for_render, 
            system_status,
            event_log_history,
            traffic_analyzer.get_all_detected_plates(), # Pass all detected plates for log
            frame_counter_for_animation,
            plate_lookup_details=current_plate_details_display # Pass details for pop-up
        )

        # --- Composite and Display ---
        frame_rgba = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        pil_combined_image = Image.alpha_composite(frame_rgba, hud_layer)

        cv2.imshow("AI-Powered Traffic Monitoring (JARVIS-Level HUD)", cv2.cvtColor(np.array(pil_combined_image), cv2.COLOR_RGBA2BGR))

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("[JARVIS-LOG] Traffic monitoring terminated. All systems offline.")


if __name__ == "__main__":
    run_traffic_monitoring()
