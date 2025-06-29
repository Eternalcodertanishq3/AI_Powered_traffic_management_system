import cv2
import numpy as np

def show_dashboard(frame, vehicle_count, human_count, state_vector,
                   scene_label="N/A", scene_confidence=0.0,
                   actions=None, metrics=None,
                   boxes_veh=None, scores_veh=None,
                   boxes_hum=None, scores_hum=None,
                   yolo_conf_threshold=0.25,
                   yolo_class_names=None): # NEW: Pass YOLO class names to draw labels
    """
    Overlays on the frame:
      - "Vehicles: X  Humans: Y"
      - "State: [...]"
      - "Actions: ..." if provided
      - "Scene: [Label] [Confidence %]"
      - Additional metrics (e.g. Frame, Epsilon, AvgReward, AvgLoss, ETA, Persons in Frame, Total Unique Persons)
      - A simple progress bar.
      - Draws bounding boxes for detected vehicles and humans with dynamic colors and labels.
    """
    window_name = "Dashboard"

    TARGET_WIDTH = 1920
    TARGET_HEIGHT = 1080
    
    resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, TARGET_WIDTH, TARGET_HEIGHT)
    cv2.moveWindow(window_name, 0, 0)


    # --- Text and Drawing Parameters (tuned for 1920x1080) ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = min(TARGET_WIDTH, TARGET_HEIGHT) * 0.001 
    
    header_font_scale = base_font_scale * 1.5
    info_font_scale = base_font_scale * 1.0
    metric_font_scale = base_font_scale * 0.8

    font_thickness_header = max(1, int(base_font_scale * 3.5))
    font_thickness_info = max(1, int(base_font_scale * 2.5))
    font_thickness_metrics = max(1, int(base_font_scale * 1.5))

    # Dynamic y_offset calculation for text placement
    y_offset = int(TARGET_HEIGHT * 0.04)

    text_color_yolo = (0, 255, 0)     # Green
    text_color_state = (0, 255, 255)  # Yellow
    text_color_actions = (0, 0, 255)  # Red
    text_color_scene = (255, 165, 0)  # Orange
    text_color_metrics = (255, 255, 255) # White

    cv2.putText(resized_frame, f"Vehicles: {vehicle_count}  Humans: {human_count}", (10, y_offset),
                font, header_font_scale, text_color_yolo, font_thickness_header)
    y_offset += int(TARGET_HEIGHT * 0.06)

    cv2.putText(resized_frame, f"State: {state_vector}", (10, y_offset),
                font, info_font_scale, text_color_state, font_thickness_info)
    y_offset += int(TARGET_HEIGHT * 0.045)

    if actions is not None:
        action_text = " | ".join([f"L{i}:{act}" for i, act in enumerate(actions)])
        cv2.putText(resized_frame, f"Actions: {action_text}", (10, y_offset),
                    font, info_font_scale, text_color_actions, font_thickness_info)
        y_offset += int(TARGET_HEIGHT * 0.045)

    cv2.putText(resized_frame, f"Scene: {scene_label} {scene_confidence:.1%}", (10, y_offset),
                font, info_font_scale, text_color_scene, font_thickness_info)
    y_offset += int(TARGET_HEIGHT * 0.055)

    if metrics is not None:
        dy = int(TARGET_HEIGHT * 0.035)
        for key, value in metrics.items():
            cv2.putText(resized_frame, f"{key}: {value}", (10, y_offset),
                        font, metric_font_scale, text_color_metrics, font_thickness_metrics)
            y_offset += dy

        if "Frame" in metrics:
            try:
                frame_metric_parts = metrics["Frame"].split("/")
                current_frame_str = frame_metric_parts[0]
                total_frames_str = frame_metric_parts[1] if len(frame_metric_parts) > 1 else '0'

                current = int(current_frame_str)
                total = int(total_frames_str) if total_frames_str.isdigit() else 0

                bar_width = int(TARGET_WIDTH * 0.2)
                bar_height = int(TARGET_HEIGHT * 0.025)
                x_start = TARGET_WIDTH - bar_width - int(TARGET_WIDTH * 0.01)
                y_start = TARGET_HEIGHT - bar_height - int(TARGET_HEIGHT * 0.01)
                
                progress = current / total if total > 0 else 0
                
                cv2.rectangle(resized_frame, (x_start, y_start), (x_start + bar_width, y_start + bar_height), (50, 50, 50), -1)
                cv2.rectangle(resized_frame, (x_start, y_start), (int(x_start + bar_width * progress), y_start + bar_height), (0, 255, 0), -1)
                cv2.rectangle(resized_frame, (x_start, y_start), (x_start + bar_width, y_start + bar_height), (200, 200, 200), 2)

                if total > 0:
                    percent_text = f"{progress:.1%}"
                else:
                    percent_text = total_frames_str

                text_size = cv2.getTextSize(percent_text, font, metric_font_scale * 1.2, font_thickness_metrics)[0]
                text_x = x_start + (bar_width - text_size[0]) // 2
                text_y = y_start + (bar_height + text_size[1]) // 2 + 2
                cv2.putText(resized_frame, percent_text, (text_x, text_y), font, metric_font_scale * 1.2, (255, 255, 255), font_thickness_metrics)

            except Exception as e:
                pass


    # --- Draw Bounding Boxes with Labels and Dynamic Colors ---
    # Determine box color based on scene classification
    box_color_normal = (0, 255, 0) # Green for normal detections
    box_color_accident = (0, 0, 255) # Red for detections during an accident scene
    
    current_box_color_veh = box_color_normal
    current_box_color_hum = box_color_normal

    # If the scene is classified as 'accident' or 'post_accident_clearance', make boxes red
    if scene_label in ["accident", "post_accident_clearance", "multi_vehicle_collision", "single_vehicle_incident", "overturned_vehicle"]:
        current_box_color_veh = box_color_accident
        current_box_color_hum = box_color_accident # Persons might be involved in accident

    scale_x = TARGET_WIDTH / frame.shape[1]
    scale_y = TARGET_HEIGHT / frame.shape[0]
    
    # Function to draw a single bounding box with label and confidence
    def draw_box_and_label(img, box, score, class_id, color, scale_x, scale_y, font, info_font_scale, font_thickness_info, class_names):
        x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, font_thickness_info)

        if class_names and class_id < len(class_names):
            label_text = f"{class_names[class_id]}: {score:.2f}"
        else:
            label_text = f"Object: {score:.2f}"

        # Calculate text size to create a background rectangle
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, info_font_scale, font_thickness_info)
        
        # Position the label above the box
        label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 5 # Avoid going off screen top
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1) # Background rectangle
        cv2.putText(img, label_text, (x1 + 5, y1 - 5), font, info_font_scale, (255, 255, 255), font_thickness_info, cv2.LINE_AA)


    # Draw vehicles
    if boxes_veh is not None and scores_veh is not None and len(boxes_veh) > 0:
        for i, box in enumerate(boxes_veh):
            if scores_veh[i] < yolo_conf_threshold:
                continue
            # Note: We need the actual class ID of the vehicle from YOLO results,
            # but currently detection_model.py only returns generic boxes_veh.
            # To draw specific vehicle types, detection_model.py would need to return
            # (boxes, scores, classes) instead of just separated vehicle/human boxes.
            # For now, we'll draw a generic 'vehicle' label.
            draw_box_and_label(resized_frame, box, scores_veh[i], -1, current_box_color_veh, # -1 for generic
                                scale_x, scale_y, font, info_font_scale, font_thickness_info, ["vehicle"])

    # Draw humans
    if boxes_hum is not None and scores_hum is not None and len(boxes_hum) > 0:
        for i, box in enumerate(boxes_hum):
            if scores_hum[i] < yolo_conf_threshold:
                continue
            draw_box_and_label(resized_frame, box, scores_hum[i], -1, current_box_color_hum, # -1 for generic
                                scale_x, scale_y, font, info_font_scale, font_thickness_info, ["person"])


    cv2.imshow(window_name, resized_frame)

