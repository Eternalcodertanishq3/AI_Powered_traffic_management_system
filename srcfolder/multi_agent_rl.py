import os
import time
import cv2
import numpy as np
import torch

from .dataset_loader import load_dataset
from .detection_model import detect_objects_and_classify_scene # Will now return more YOLO data
from .video_processing import video_stream_from_dataset
from .traffic_state import TrafficState
from .multi_agent_rl import MultiAgent
from .dashboard import show_dashboard
from .constants import SCENE_CLASSES, PERSON_CLASS_ID, VEHICLE_CLASS_IDS # Import for robust checking
from .person_tracker import SimplePersonTracker

def linear_epsilon(step, eps_start=1.0, eps_end=0.05, decay_steps=5000):
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)

def main():
    # --- GLOBAL CONFIGURATION PARAMETERS ──────────────────────────────────
    # Dataset Configuration (used only if VIDEO_INPUT_SOURCE is "DATASET")
    DATASET_ROOT_DIR = r"C:/Personal Projects/AI_Powered_traffic_management_system/data/trafficnet_dataset_v1/trafficnet_dataset_v1/train"
    DATASET_SUBFOLDER = "image_2"
    DATASET_EXTENSION = "jpg"

    # YOLO Object Detection Confidence Threshold
    # Lower value (e.g., 0.1-0.3): Detects more objects, potentially more false positives.
    # Higher value (e.g., 0.7-0.9): Detects fewer objects, but generally more accurate.
    YOLO_DETECTION_CONFIDENCE = 0.25

    # VIDEO INPUT SOURCE CONFIGURATION ──────────────────────────────────
    # UNCOMMENT only ONE of the following options:

    # Option A: Use a pre-existing image dataset (configured above)
    # VIDEO_INPUT_SOURCE = "DATASET"

    # Option B: Use a live webcam feed
    # VIDEO_INPUT_SOURCE = "WEBCAM"

    # Option C: Use a video file (MP4, AVI, etc.)
    # Provide the FULL PATH to your video file.
    VIDEO_INPUT_SOURCE = r"C:/Personal Projects/AI_Powered_traffic_management_system/data/1643-148614430.mp4"
    # To test the accident video you provided:
    # VIDEO_INPUT_SOURCE = r"C:/Personal Projects/AI_Powered_traffic_management_system/data/gettyimages-1936679257-640_adpp.mp4"

    # --- END OF GLOBAL CONFIGURATION PARAMETERS ──────────────────────────

    # Dynamic generation of a unique identifier for model checkpoints
    if VIDEO_INPUT_SOURCE == "DATASET":
        dataset_path_for_id = DATASET_ROOT_DIR
    else:
        dataset_path_for_id = "live_feed" if VIDEO_INPUT_SOURCE == "WEBCAM" else os.path.basename(VIDEO_INPUT_SOURCE).split('.')[0]

    dataset_identifier_parts = [part for part in dataset_path_for_id.split('/') if part]
    if len(dataset_identifier_parts) > 2:
        dataset_identifier = "_".join(dataset_identifier_parts[-2:])
    else:
        dataset_identifier = dataset_identifier_parts[-1] if dataset_identifier_parts else "default_session"

    dataset_identifier = dataset_identifier.replace(":", "").replace("\\", "_").replace("/", "_").lower()

    MODEL_SAVE_DIR = os.path.join("models", dataset_identifier)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"Model checkpoints will be saved/loaded to/from: {MODEL_SAVE_DIR}")

    total_frames = 0
    stream = None
    cap = None

    if VIDEO_INPUT_SOURCE == "DATASET":
        try:
            dataset = load_dataset(
                root_dir=DATASET_ROOT_DIR,
                subfolder=DATASET_SUBFOLDER,
                extension=DATASET_EXTENSION
            )
            print(f"Number of images in dataset: {len(dataset)}")
            total_frames = len(dataset)
            stream = video_stream_from_dataset(dataset)
        except FileNotFoundError as e:
            print(f"[CRITICAL ERROR] Dataset not found: {e}. Please check DATASET_ROOT_DIR. Exiting.")
            return
    else:
        print(f"Initializing video capture from: '{VIDEO_INPUT_SOURCE}'")
        cap_source = 0 if VIDEO_INPUT_SOURCE == "WEBCAM" else VIDEO_INPUT_SOURCE
        cap = cv2.VideoCapture(cap_source)
        
        if not cap.isOpened():
            print(f"[CRITICAL ERROR] Could not open video source '{cap_source}'. Please check the path or webcam connection. Exiting.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(cap_source, str) else 0
        print(f"Total frames (estimated for file, 0 for live webcam): {total_frames}")

        def cv2_video_stream_generator(capture_obj):
            while True:
                ret, frame = capture_obj.read()
                if not ret:
                    break
                yield frame
        stream = cv2_video_stream_generator(cap)

    if stream is None:
        print("[CRITICAL ERROR] Video stream could not be initialized. Exiting.")
        return

    traffic_state = TrafficState(num_lanes=4)
    agent_kwargs = dict(
        state_dim = 2,
        action_dim = 2,
        lr = 1e-3,
        gamma = 0.99,
        buffer_size = 20000,
        batch_size = 128,
        target_update = 1000
    )
    multi_agent = MultiAgent(num_agents=4, **agent_kwargs)

    for idx, agent in enumerate(multi_agent.agents):
        ckpt_path_agent = os.path.join(MODEL_SAVE_DIR, f"agent_lane_{idx}.pth")
        if os.path.exists(ckpt_path_agent):
            try:
                sd = torch.load(ckpt_path_agent, map_location=agent.device)
                agent.model.load_state_dict(sd)
                agent.target_model.load_state_dict(agent.model.state_dict())
                print(f"[RESUME] Loaded RL agent {idx} from '{ckpt_path_agent}'")
            except RuntimeError as e:
                print(f"[SKIP] Checkpoint for RL agent {idx} in '{MODEL_SAVE_DIR}' incompatible. Starting fresh. ({e})")
            except Exception as e:
                print(f"[ERROR] Could not load RL agent {idx} checkpoint from '{ckpt_path_agent}': {e}. Starting fresh.")
        else:
            print(f"[INFO] No checkpoint found for RL agent {idx} in '{MODEL_SAVE_DIR}'. Starting fresh.")

    start_time = time.time()
    frame_count = 0
    reward_hist = []
    loss_hist = []

    person_tracker = SimplePersonTracker(iou_threshold=0.3, max_unmatched_frames=10)

    # Read the first frame to ensure window setup and initial display
    try:
        initial_frame = next(stream)
        if initial_frame is None:
             print("[CRITICAL ERROR] Initial frame is None. Video stream likely empty or problematic. Exiting.")
             return
    except StopIteration:
        print("[CRITICAL ERROR] Video stream empty from the start. Exiting.")
        return
    except Exception as e:
        print(f"[CRITICAL ERROR] Error getting initial frame: {e}. Exiting.")
        return

    # Call detection with increased return values for all YOLO objects and class names
    vc, hc, all_boxes, all_scores, all_classes, scene_label, scene_confidence, yolo_class_names = \
        detect_objects_and_classify_scene(initial_frame, conf_threshold=YOLO_DETECTION_CONFIDENCE)
    
    # Filter only person boxes for the person tracker
    initial_person_boxes = all_boxes[all_classes == PERSON_CLASS_ID]
    person_tracker.update(initial_person_boxes)
    current_person_count = len(person_tracker.tracked_persons)

    traffic_state.update_state([vc // 4] * 4)
    initial_states = [[traffic_state.lane_counts[i], traffic_state.signal_state[i]] for i in range(4)]
    initial_actions = multi_agent.select_actions(initial_states, linear_epsilon(1))
    
    show_dashboard(
        initial_frame, vc, hc, traffic_state.get_state_vector(),
        scene_label=scene_label, scene_confidence=scene_confidence,
        actions=initial_actions, 
        metrics={"Frame": "1/Live" if total_frames == 0 else f"1/{total_frames}",
                 "Persons in Frame": current_person_count,
                 "Total Unique Persons": person_tracker.get_total_unique_persons()},
        boxes_veh=all_boxes[np.isin(all_classes, list(VEHICLE_CLASS_IDS))], # Filter vehicle boxes for dashboard
        scores_veh=all_scores[np.isin(all_classes, list(VEHICLE_CLASS_IDS))],
        boxes_hum=all_boxes[all_classes == PERSON_CLASS_ID], # Filter human boxes for dashboard
        scores_hum=all_scores[all_classes == PERSON_CLASS_ID],
        yolo_conf_threshold=YOLO_DETECTION_CONFIDENCE,
        yolo_class_names=yolo_class_names # Pass YOLO's class names
    )
    frame_count = 1

    # Main loop for subsequent frames
    while True:
        try:
            frame = next(stream)
        except StopIteration:
            print("End of video stream.")
            break
        except Exception as e:
            print(f"Error getting frame: {e}. Stopping stream.")
            break

        frame_count += 1

        # Call detection with increased return values for all YOLO objects and class names
        vc, hc, all_boxes, all_scores, all_classes, scene_label, scene_confidence, yolo_class_names = \
            detect_objects_and_classify_scene(frame, conf_threshold=YOLO_DETECTION_CONFIDENCE)

        # Update person tracker with current human detections
        current_person_boxes = all_boxes[all_classes == PERSON_CLASS_ID]
        person_tracker.update(current_person_boxes)
        current_person_count = len(person_tracker.tracked_persons)
        total_unique_persons = person_tracker.get_total_unique_persons()

        # --- Emergency Vehicle Detection and Logic (Conceptual) ---
        # This part assumes you've trained a custom YOLO model to detect 'ambulance'
        # and possibly infer its direction.
        
        # Placeholder variables - these would come from your custom YOLO and tracking
        is_emergency_vehicle_present = False 
        emergency_vehicle_direction = None # e.g., 'North', 'East', 'South', 'West'
        
        # --- Emergency Vehicle Detection Logic ---
        # Find 'emergency_vehicle' detections if you have a custom YOLO model
        # For example, if 'emergency_vehicle' is class ID 80 in your custom model:
        # if yolo_class_names and 'emergency_vehicle' in yolo_class_names:
        #     emergency_vehicle_class_id = yolo_class_names.index('emergency_vehicle')
        #     emergency_vehicle_detections_mask = (all_classes == emergency_vehicle_class_id)
        #     if np.any(emergency_vehicle_detections_mask):
        #         is_emergency_vehicle_present = True
        #         # Further logic to infer direction from boxes_veh or a dedicated tracker
        #         # For example, rudimentary direction based on relative position to center of frame/intersection
        #         # This requires intersection coordinates and more complex tracking
        #         # emergency_vehicle_direction = infer_direction(all_boxes[emergency_vehicle_detections_mask][0])


        # --- Dynamic Traffic Light Control Logic (Conceptual) ---
        # This would override or heavily influence the RL agent's actions if an emergency vehicle is present.
        rl_actions_override = False
        if is_emergency_vehicle_present and emergency_vehicle_direction:
            # print(f"Emergency vehicle detected from {emergency_vehicle_direction}! Prioritizing traffic.")
            # Based on `emergency_vehicle_direction`, determine which traffic lights to open/close.
            # This requires mapping directions to lanes (e.g., North approach -> Lane 0)
            # and logic to set conflicting lanes to red.
            
            # Example: Assuming Lane 0 is North-bound, Lane 1 East, Lane 2 South, Lane 3 West
            # if emergency_vehicle_direction == 'North':
            #     traffic_state.signal_state[0] = 1 # Force North Green
            #     traffic_state.signal_state[1] = 0 # Force East Red
            #     traffic_state.signal_state[2] = 0 # Force South Red
            #     traffic_state.signal_state[3] = 0 # Force West Red
            #     rl_actions_override = True # Indicate that RL actions were overridden

            pass # Placeholder for actual emergency logic


        lane_counts = [vc // 4] * 4
        traffic_state.update_state(lane_counts)

        states = [[traffic_state.lane_counts[i], traffic_state.signal_state[i]] for i in range(4)]

        eps = linear_epsilon(frame_count)
        actions = multi_agent.select_actions(states, eps) # RL selects actions

        # If emergency vehicle logic activated, use its forced actions instead of RL's
        # if rl_actions_override:
        #     # 'actions' here would be the emergency-forced actions
        #     pass 

        for i, a in enumerate(actions):
            if a == 1:
                traffic_state.signal_state[i] ^= 1

        next_states = [[traffic_state.lane_counts[i], traffic_state.signal_state[i]] for i in range(4)]
        rewards = [-traffic_state.lane_counts[i] for i in range(4)]
        dones = [False] * 4

        # --- Reward Function Modification for Emergency Vehicles (Conceptual) ---
        # If an emergency vehicle is present, significantly alter rewards to prioritize it.
        # This part would typically be inside multi_agent_rl.py's reward calculation.
        # Example (if emergency_vehicle_direction is known and a specific lane is affected):
        # if is_emergency_vehicle_present:
        #     emergency_lane_idx = 0 # Map direction to lane index
        #     if traffic_state.signal_state[emergency_lane_idx] == 1: # If green
        #         rewards[emergency_lane_idx] += 1000 # Huge positive reward
        #     else: # If red
        #         rewards[emergency_lane_idx] -= 1000 # Huge negative penalty


        multi_agent.push(states, actions, rewards, next_states, dones)
        loss = multi_agent.update()

        reward_hist.append(np.mean(rewards))
        if loss is not None:
            loss_hist.append(loss)
        
        elapsed = time.time() - start_time
        avg_reward = np.mean(reward_hist[-100:]) if len(reward_hist) >= 100 else np.mean(reward_hist)
        avg_loss = np.mean(loss_hist[-100:]) if len(loss_hist) >= 100 else (loss_hist[-1] if loss_hist else 0)
        
        remaining_time = (total_frames - frame_count) * (elapsed / frame_count) if total_frames > 0 and frame_count > 0 else 0

        metrics = {
            "Frame": f"{frame_count}/{total_frames if total_frames > 0 else 'Live'}",
            "Epsilon": f"{eps:.2f}",
            "AvgReward": f"{avg_reward:.2f}",
            "AvgLoss": f"{avg_loss:.4f}",
            "Persons in Frame": current_person_count,
            "Total Unique Persons": total_unique_persons,
            "ETA": f"{remaining_time/60:.1f}m" if total_frames > 0 else "N/A"
        }

        # Pass all necessary data to show_dashboard for processing and display
        show_dashboard(
            frame, vc, hc, traffic_state.get_state_vector(),
            scene_label=scene_label, scene_confidence=scene_confidence,
            actions=actions, metrics=metrics,
            # Pass ALL YOLO detections, the dashboard will filter/draw them.
            boxes_veh=all_boxes, # Now contains all detected objects, not just vehicles/humans
            scores_veh=all_scores,
            boxes_hum=all_classes, # Pass classes too so dashboard can use them
            yolo_conf_threshold=YOLO_DETECTION_CONFIDENCE,
            yolo_class_names=yolo_class_names # Pass YOLO's internal class names for labels
        )

        if frame_count % 500 == 0:
            print(f"[{frame_count:05d}/{total_frames if total_frames > 0 else 'Live'}] Epsilon={eps:.2f}  AvgReward={avg_reward:.2f}  AvgLoss={avg_loss:.4f}  ETA={metrics['ETA']}")
            for idx, agent in enumerate(multi_agent.agents):
                torch.save(agent.model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"agent_lane_{idx}.pth"))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested quit. Exiting application.")
            break

    cv2.destroyAllWindows()
    if cap is not None and cap.isOpened():
        cap.release()
    total_time = time.time() - start_time
    print(f"Done: Processed {frame_count} frames in {total_time/60:.2f} minutes.")
    print(f"All RL models saved to: {MODEL_SAVE_DIR}")
    print(f"Final count of total unique persons detected and exited/lost: {person_tracker.get_total_unique_persons()}")

if __name__ == "__main__":
    main()

