# srcfolder/traffic_analysis.py

import math
import random
import time
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

# Import constants
from .constants import (
    IOU_THRESHOLD_FOR_TRACKING, MAX_TRACKING_AGE, CONFIDENCE_SMOOTHING_WINDOW_SIZE,
    MIN_CONF_FOR_SMOOTHING_HISTORY, TRAJECTORY_PREDICTION_LENGTH, TRAJECTORY_SMOOTHING_FACTOR,
    THREAT_BASE_SCORES, THREAT_SPEED_MULTIPLIER, THREAT_DISTANCE_INVERSE_MULTIPLIER,
    BRIGHTNESS_CHANGE_THRESHOLD, DENSITY_CHANGE_THRESHOLD, ENVIRONMENTAL_ALERT_COOLDOWN,
    NPR_SIMULATION_CHANCE, PLATE_LOG_MAX_SIZE,
    COLLISION_PROXIMITY_THRESHOLD, COLLISION_ANGLE_THRESHOLD, COLLISION_ALERT_COOLDOWN,
    TRAFFIC_ZONE_COUNT, TRAFFIC_FLOW_SPEED_THRESHOLD_SLOWDOWN, TRAFFIC_FLOW_ALERT_COOLDOWN,
    TAILGATING_DISTANCE_THRESHOLD, TAILGATING_SPEED_DIFF_THRESHOLD, AGGRESSIVE_LANE_CHANGE_THRESHOLD,
    DRIVER_BEHAVIOR_ALERT_COOLDOWN,
    SCENE_CLASSES, ACCIDENT_CONFIDENCE_THRESHOLD_OBSERVE, ACCIDENT_CONFIDENCE_THRESHOLD_WARN,
    ACCIDENT_CONFIDENCE_THRESHOLD_CRITICAL_ALERT, ACCIDENT_PERSISTENCE_FRAMES_WARN,
    ACCIDENT_PERSISTENCE_FRAMES_CRITICAL, ALERT_COOLDOWN_SECONDS,
    DENSE_TRAFFIC_CAR_COUNT_THRESHOLD_FOR_FALSE_POSITIVE, ACCIDENT_IMPACT_OBJECTS,
    MIN_IMPACT_OBJECTS_FOR_ACCIDENT,
    SCENE_SMOOTHING_WINDOW_SIZE,
    MIN_SCENE_CONFIDENCE_DISPLAY
)

class TrafficAnalyzer:
    def __init__(self, frame_width: int, frame_height: int, global_ui_scale: float):
        self.tracked_objects: Dict[int, Dict[str, Any]] = {}
        self.next_object_id = 0
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.global_ui_scale = global_ui_scale

        self.prev_avg_brightness = -1
        self.prev_object_density = -1
        self.last_brightness_alert_time = 0
        self.last_density_alert_time = 0
        self.last_collision_alert_time = 0
        self.last_traffic_flow_alert_time = 0
        self.last_driver_behavior_alert_time = 0

        self.traffic_zone_speeds = [deque(maxlen=10) for _ in range(TRAFFIC_ZONE_COUNT)]
        self.all_detected_plates = deque(maxlen=PLATE_LOG_MAX_SIZE)
        self.simulated_plate_data: Dict[str, Dict[str, str]] = {}

        # For scene analysis and alerts
        self.scene_prediction_history = deque(maxlen=SCENE_SMOOTHING_WINDOW_SIZE)
        self.current_alert_level = "OBSERVATION"
        self.consecutive_accident_frames = 0
        self.last_alert_timestamp = 0

    def _calculate_iou(self, box1, box2):
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

    def _generate_simulated_plate_data(self, plate_num: str) -> Dict[str, str]:
        """Generates simulated data for a given number plate."""
        if plate_num not in self.simulated_plate_data:
            owner_names = ["John Doe", "Jane Smith", "Robert Johnson", "Emily Davis", "Michael Brown"]
            vehicle_types = ["Sedan", "SUV", "Hatchback", "Truck", "Minivan", "Motorcycle"]
            statuses = ["Registered", "Expired (Alert)", "Stolen (CRITICAL)", "Pending"]
            colors = ["Red", "Blue", "Black", "White", "Silver", "Grey", "Green", "Yellow"]
            makes = ["Generic Motors", "Universal Auto", "Apex Vehicles", "Global Trans"]
            
            owner = random.choice(owner_names)
            vehicle = random.choice(vehicle_types)
            status = random.choice(statuses)
            color = random.choice(colors)
            make = random.choice(makes)

            self.simulated_plate_data[plate_num] = {
                "Owner": owner,
                "Vehicle Type": vehicle,
                "Color": color,
                "Make": make,
                "Registration Status": status,
                "Last Scanned": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        return self.simulated_plate_data[plate_num]

    def _generate_vehicle_signature(self, label: str) -> Dict[str, str]:
        """Generates a simulated vehicle signature."""
        colors = ["Red", "Blue", "Black", "White", "Silver", "Grey", "Green", "Yellow"]
        makes_car = ["Generic Sedan", "Universal SUV", "Apex Hatchback"]
        makes_truck_bus = ["Global Trucking", "Heavy Haul Co.", "City Transit"]

        sig = {"Type": label.capitalize()}
        sig["Color"] = random.choice(colors)
        if label == "car":
            sig["Make"] = random.choice(makes_car)
        elif label in ["truck", "bus"]:
            sig["Make"] = random.choice(makes_truck_bus)
        else:
            sig["Make"] = "N/A"
        return sig

    def update_object_tracking(self, detections: List[Dict[str, Any]], frame_count: int) -> List[Dict[str, Any]]:
        """
        Updates tracked objects based on current frame detections (Feature 1 & 2).
        Returns a list of objects ready for display.
        """
        matched_track_ids = set()
        
        # Update existing tracks
        for track_id, track_obj in list(self.tracked_objects.items()): 
            matched_this_frame = False
            best_iou = 0.0
            best_det_idx = -1

            for i, det in enumerate(detections):
                if det['label'] != track_obj['label']:
                    continue

                iou = self._calculate_iou(track_obj['bbox'], det['bbox'])
                if iou > best_iou and iou > IOU_THRESHOLD_FOR_TRACKING:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx != -1:
                det = detections.pop(best_det_idx) # Remove matched detection
                
                cx_new = (det['bbox'][0] + det['bbox'][2]) / 2
                cy_new = (det['bbox'][1] + det['bbox'][3]) / 2
                
                if 'prev_bbox_center' in track_obj and track_obj['prev_bbox_center'] is not None:
                    cx_prev, cy_prev = track_obj['prev_bbox_center']
                    vx = cx_new - cx_prev
                    vy = cy_new - cy_prev
                    
                    track_obj['velocity_x'] = track_obj.get('velocity_x', 0) * (1 - TRAJECTORY_SMOOTHING_FACTOR) + vx * TRAJECTORY_SMOOTHING_FACTOR
                    track_obj['velocity_y'] = track_obj.get('velocity_y', 0) * (1 - TRAJECTORY_SMOOTHING_FACTOR) + vy * TRAJECTORY_SMOOTHING_FACTOR
                    
                    if 'horizontal_movement_history' not in track_obj:
                        track_obj['horizontal_movement_history'] = deque(maxlen=5)
                    track_obj['horizontal_movement_history'].append(abs(vx))

                else:
                    track_obj['velocity_x'] = 0.0
                    track_obj['velocity_y'] = 0.0
                    track_obj['horizontal_movement_history'] = deque(maxlen=5)


                track_obj['bbox'] = det['bbox'] 
                track_obj['last_seen'] = frame_count
                track_obj['prev_bbox_center'] = (cx_new, cy_new)

                if det['confidence'] >= MIN_CONF_FOR_SMOOTHING_HISTORY: 
                    track_obj['confidence_history'].append(det['confidence'])
                matched_track_ids.add(track_id)
                matched_this_frame = True
            
            if not matched_this_frame:
                track_obj['last_seen_unmatched_count'] = track_obj.get('last_seen_unmatched_count', 0) + 1
                # Update prev_bbox_center even if not matched to allow continued trajectory calc
                cx_current = (track_obj['bbox'][0] + track_obj['bbox'][2]) / 2
                cy_current = (track_obj['bbox'][1] + track_obj['bbox'][3]) / 2
                track_obj['prev_bbox_center'] = (cx_current, cy_current)
            else:
                track_obj['last_seen_unmatched_count'] = 0 


        # Add new detections as new tracks
        for det in detections: 
            if det['confidence'] >= MIN_CONF_FOR_SMOOTHING_HISTORY: 
                cx_new = (det['bbox'][0] + det['bbox'][2]) / 2
                cy_new = (det['bbox'][1] + det['bbox'][3]) / 2

                plate_num = None
                plate_data = {}
                if det['label'] in ["car", "truck"]:
                    if random.random() < NPR_SIMULATION_CHANCE: 
                        plate_num = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=7))
                        plate_data = self._generate_simulated_plate_data(plate_num)
                        if plate_num not in self.all_detected_plates:
                            self.all_detected_plates.append(plate_num)
                            print(f"[PLATE LOG] New Plate Detected: {plate_num}") # Print to terminal

                self.tracked_objects[self.next_object_id] = {
                    'id': self.next_object_id,
                    'bbox': det['bbox'],
                    'label': det['label'],
                    'last_seen': frame_count,
                    'last_seen_unmatched_count': 0,
                    'confidence_history': deque([det['confidence']], maxlen=CONFIDENCE_SMOOTHING_WINDOW_SIZE),
                    'velocity_x': 0.0, 
                    'velocity_y': 0.0,
                    'prev_bbox_center': (cx_new, cy_new),
                    'threat_score': 0.0,
                    'plate_number': plate_num, 
                    'plate_data': plate_data,
                    'vehicle_signature': self._generate_vehicle_signature(det['label']), # Feature 9
                    'is_collision_alert': False,
                    'horizontal_movement_history': deque(maxlen=5)
                }
                self.next_object_id += 1
        
        # Remove old tracks
        tracks_to_delete = [track_id for track_id, track_obj in self.tracked_objects.items() if track_obj['last_seen_unmatched_count'] > MAX_TRACKING_AGE]
        for track_id in tracks_to_delete:
            del self.tracked_objects[track_id]

        # Prepare objects for display with smoothed confidence and threat score
        display_objects = []
        for track_id, track_obj in self.tracked_objects.items():
            if not track_obj['confidence_history']: 
                continue
            smoothed_conf = sum(track_obj['confidence_history']) / len(track_obj['confidence_history'])
            
            if smoothed_conf >= MIN_CONF_FOR_SMOOTHING_HISTORY: # Use min conf for smoothing history for display
                # Calculate Threat Score (Feature 5)
                threat_score = THREAT_BASE_SCORES.get(track_obj['label'], 1) 
                
                speed_px_per_frame = math.sqrt(track_obj['velocity_x']**2 + track_obj['velocity_y']**2)
                threat_score += speed_px_per_frame * THREAT_SPEED_MULTIPLIER

                bbox_area = (track_obj['bbox'][2] - track_obj['bbox'][0]) * (track_obj['bbox'][3] - track_obj['bbox'][1])
                relative_distance_val = 0
                if bbox_area > 0:
                    relative_distance_val = 1.0 / (math.sqrt(bbox_area) / (self.frame_width * self.global_ui_scale) + 0.0001)
                
                threat_score += (THREAT_DISTANCE_INVERSE_MULTIPLIER / (relative_distance_val + 1)) 
                
                track_obj['threat_score'] = min(threat_score, 100.0) 

                display_objects.append({
                    'id': track_id, 
                    'bbox': track_obj['bbox'],
                    'label': track_obj['label'],
                    'confidence': smoothed_conf,
                    'velocity_x': track_obj['velocity_x'],
                    'velocity_y': track_obj['velocity_y'],
                    'threat_score': track_obj['threat_score'],
                    'plate_number': track_obj['plate_number'], 
                    'plate_data': track_obj['plate_data'],
                    'vehicle_signature': track_obj['vehicle_signature'], # Feature 9
                    'is_collision_alert': track_obj.get('is_collision_alert', False),
                    'horizontal_movement_history': track_obj['horizontal_movement_history']
                })
        return display_objects

    def analyze_scene_and_alerts(self, scene_report: Dict[str, Any], event_log_history: deque) -> Tuple[str, str, str]:
        """
        Analyzes scene prediction and manages alert levels.
        Returns (current_alert_level, display_action_message, most_common_scene)
        """
        current_time = time.time()
        current_scene_label = scene_report["main_prediction"]
        scene_confidence = scene_report["main_confidence"]

        if scene_confidence >= MIN_SCENE_CONFIDENCE_DISPLAY:
            self.scene_prediction_history.append(current_scene_label)

        most_common_scene = "N/A"
        smoothed_scene_confidence = 0.0
        if self.scene_prediction_history:
            label_counts = {}
            for label in self.scene_prediction_history:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            if label_counts:
                most_common_scene = max(label_counts, key=label_counts.get)
                smoothed_scene_confidence = label_counts[most_common_scene] / len(self.scene_prediction_history)

        # Count current cars and accident impact objects from tracked objects
        current_cars = sum(1 for obj in self.tracked_objects.values() if obj['label'] in ["car", "truck", "bus"])
        current_accident_impact_objects = sum(1 for obj in self.tracked_objects.values() if obj['label'] in ACCIDENT_IMPACT_OBJECTS)

        force_observation = False
        if most_common_scene == "dense_traffic" and smoothed_scene_confidence > 0.8:
            force_observation = True
            if self.current_alert_level != "OBSERVATION":
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Override: Dense Traffic Confirmed.")
        elif most_common_scene == "accident" and current_cars > DENSE_TRAFFIC_CAR_COUNT_THRESHOLD_FOR_FALSE_POSITIVE and current_accident_impact_objects < MIN_IMPACT_OBJECTS_FOR_ACCIDENT:
            force_observation = True
            if self.current_alert_level != "OBSERVATION":
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Anomaly: Accident-like, but dense traffic. Verification needed.")
        
        if self.current_alert_level != "OBSERVATION" and \
           (most_common_scene != "accident" or force_observation) and \
           (current_time - self.last_alert_timestamp) > ALERT_COOLDOWN_SECONDS:
            self.current_alert_level = "OBSERVATION"
            self.consecutive_accident_frames = 0
            event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Status: Monitoring (Resolved/Cleared)")
        
        if not force_observation and most_common_scene == "accident" and smoothed_scene_confidence >= ACCIDENT_CONFIDENCE_THRESHOLD_OBSERVE:
            self.consecutive_accident_frames += 1
            if self.consecutive_accident_frames >= ACCIDENT_PERSISTENCE_FRAMES_CRITICAL and smoothed_scene_confidence >= ACCIDENT_CONFIDENCE_THRESHOLD_CRITICAL_ALERT:
                if self.current_alert_level != "ALERT_SENT": 
                    self.current_alert_level = "CRITICAL_ALERT"
                    self.last_alert_timestamp = current_time 
                    print(f"[JARVIS-ALERT] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - CRITICAL ACCIDENT DETECTED! Triggering high-priority alert system.")
                    event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - CRITICAL ACCIDENT! (Conf: {smoothed_scene_confidence:.2f})")
            elif self.consecutive_accident_frames >= ACCIDENT_PERSISTENCE_FRAMES_WARN and smoothed_scene_confidence >= ACCIDENT_CONFIDENCE_THRESHOLD_WARN:
                self.current_alert_level = "WARNING"
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Potential Incident. (Conf: {smoothed_scene_confidence:.2f})")
            else:
                self.current_alert_level = "OBSERVATION"
        elif force_observation: 
            self.current_alert_level = "OBSERVATION"
            self.consecutive_accident_frames = 0
        else: 
            self.consecutive_accident_frames = 0
            self.current_alert_level = "OBSERVATION" 
            

        display_action_message = scene_report["suggested_action"]
        if self.current_alert_level == "WARNING":
            display_action_message = f"VERIFICATION REQUIRED: Potential Incident. (Conf: {smoothed_scene_confidence:.2f})"
        elif self.current_alert_level == "CRITICAL_ALERT":
            display_action_message = f"URGENT: DISPATCHING EMERGENCY SERVICES! Conf: {smoothed_scene_confidence:.2f}"
            if self.last_alert_timestamp and (current_time - self.last_alert_timestamp < 5): 
                pass 
            else:
                print(f"[JARVIS-ALERT] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - CRITICAL ACCIDENT DETECTED! Alerting emergency services for scene: {most_common_scene.upper()} (Confidence: {smoothed_scene_confidence:.2f})")
                self.last_alert_timestamp = current_time
        elif self.current_alert_level == "ALERT_SENT":
            display_action_message = "ALERT DISPATCHED. Monitoring Scene for Updates."
            
        return self.current_alert_level, display_action_message, most_common_scene, smoothed_scene_confidence, scene_report["all_predictions"]

    def analyze_environmental_anomalies(self, frame: np.ndarray, event_log_history: deque):
        """Analyzes environmental anomalies like brightness and density (Feature 6)."""
        current_time = time.time()
        
        current_avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if self.prev_avg_brightness != -1 and abs(current_avg_brightness - self.prev_avg_brightness) > BRIGHTNESS_CHANGE_THRESHOLD:
            if (current_time - self.last_brightness_alert_time) > ENVIRONMENTAL_ALERT_COOLDOWN:
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Env Anomaly: Sudden Brightness Change ({current_avg_brightness:.1f})")
                self.last_brightness_alert_time = current_time
        self.prev_avg_brightness = current_avg_brightness

        density_roi_x1 = int(self.frame_width * 0.4)
        density_roi_y1 = int(self.frame_height * 0.4)
        density_roi_x2 = int(self.frame_width * 0.6)
        density_roi_y2 = int(self.frame_height * 0.6)
        density_roi_area = (density_roi_x2 - density_roi_x1) * (density_roi_y2 - density_roi_y1)
        
        objects_in_density_roi = 0
        for obj in self.tracked_objects.values():
            bbox = obj['bbox']
            if not (bbox[0] > density_roi_x2 or bbox[2] < density_roi_x1 or bbox[1] > density_roi_y2 or bbox[3] < density_roi_y1):
                objects_in_density_roi += 1
        
        current_object_density = objects_in_density_roi / (density_roi_area / (self.frame_width * self.frame_height) + 0.0001) 
        
        if self.prev_object_density != -1 and abs(current_object_density - self.prev_object_density) > DENSITY_CHANGE_THRESHOLD:
            if (current_time - self.last_density_alert_time) > ENVIRONMENTAL_ALERT_COOLDOWN:
                event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Env Anomaly: Sudden Density Change ({current_object_density:.2f})")
                self.last_density_alert_time = current_time
        self.prev_object_density = current_object_density

    def analyze_predictive_incidents(self, event_log_history: deque):
        """Analyzes for predictive incidents like collisions (Feature 10)."""
        current_time = time.time()
        if (current_time - self.last_collision_alert_time) > COLLISION_ALERT_COOLDOWN:
            # Reset collision alerts
            for tracked_obj in self.tracked_objects.values():
                tracked_obj['is_collision_alert'] = False

            for i, obj1 in enumerate(self.tracked_objects.values()):
                for j, obj2 in enumerate(self.tracked_objects.values()):
                    if obj1['id'] >= obj2['id']: continue # Avoid self-comparison and duplicate pairs

                    if obj1['label'] not in ["car", "truck", "bus", "person", "motorcycle"] or \
                       obj2['label'] not in ["car", "truck", "bus", "person", "motorcycle"]:
                        continue

                    center1_x = (obj1['bbox'][0] + obj1['bbox'][2]) / 2
                    center1_y = (obj1['bbox'][1] + obj1['bbox'][3]) / 2
                    center2_x = (obj2['bbox'][0] + obj2['bbox'][2]) / 2
                    center2_y = (obj2['bbox'][1] + obj2['bbox'][3]) / 2

                    distance = math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

                    if distance < COLLISION_PROXIMITY_THRESHOLD * self.global_ui_scale:
                        vec1_x, vec1_y = obj1['velocity_x'], obj1['velocity_y']
                        vec2_x, vec2_y = obj2['velocity_x'], obj2['velocity_y']

                        rel_vx = vec1_x - vec2_x
                        rel_vy = vec1_y - vec2_y

                        target_vec_x = center1_x - center2_x
                        target_vec_y = center1_y - center2_y

                        dot_product = rel_vx * target_vec_x + rel_vy * target_vec_y
                        
                        if dot_product < 0: # Moving towards each other
                            event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - PREDICTIVE ALERT: Collision imminent between ID:{obj1['id']} ({obj1['label']}) and ID:{obj2['id']} ({obj2['label']})!")
                            self.last_collision_alert_time = current_time
                            obj1['is_collision_alert'] = True
                            obj2['is_collision_alert'] = True
                            return # Only one collision alert per cycle for simplicity

    def analyze_traffic_flow(self, event_log_history: deque):
        """Analyzes overall traffic flow in zones (Feature 11)."""
        current_time = time.time()
        if (current_time - self.last_traffic_flow_alert_time) > TRAFFIC_FLOW_ALERT_COOLDOWN:
            zone_height = self.frame_height // TRAFFIC_ZONE_COUNT
            for i in range(TRAFFIC_ZONE_COUNT):
                zone_y1 = i * zone_height
                zone_y2 = (i + 1) * zone_height
                
                zone_objects_speeds = []
                for obj in self.tracked_objects.values():
                    if obj['label'] in ["car", "truck", "bus"]:
                        obj_center_y = (obj['bbox'][1] + obj['bbox'][3]) / 2
                        if zone_y1 <= obj_center_y < zone_y2:
                            speed = math.sqrt(obj['velocity_x']**2 + obj['velocity_y']**2)
                            if speed > 0: # Only consider moving vehicles
                                zone_objects_speeds.append(speed)
                
                if zone_objects_speeds:
                    avg_zone_speed = sum(zone_objects_speeds) / len(zone_objects_speeds)
                    self.traffic_zone_speeds[i].append(avg_zone_speed)

                    if len(self.traffic_zone_speeds[i]) > 1:
                        # Compare current avg speed to the average of the historical speeds in the deque
                        historical_avg_speed = sum(self.traffic_zone_speeds[i]) / len(self.traffic_zone_speeds[i])
                        
                        if historical_avg_speed > 0 and \
                           (historical_avg_speed - avg_zone_speed) / historical_avg_speed > TRAFFIC_FLOW_SPEED_THRESHOLD_SLOWDOWN:
                            event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Traffic Anomaly: Slowdown in Zone {i+1} ({avg_zone_speed:.1f}p/f)")
                            self.last_traffic_flow_alert_time = current_time
                            return # Only one traffic flow alert per cycle for simplicity

    def analyze_driver_behavior(self, event_log_history: deque):
        """Simulates detection of risky driver behaviors (Feature 12)."""
        current_time = time.time()
        if (current_time - self.last_driver_behavior_alert_time) > DRIVER_BEHAVIOR_ALERT_COOLDOWN:
            for i, obj1 in enumerate(self.tracked_objects.values()):
                for j, obj2 in enumerate(self.tracked_objects.values()):
                    if obj1['id'] == obj2['id']: continue

                    # Tailgating check (car/truck/bus only)
                    if obj1['label'] in ["car", "truck", "bus"] and obj2['label'] in ["car", "truck", "bus"]:
                        center1_x = (obj1['bbox'][0] + obj1['bbox'][2]) / 2
                        center1_y = (obj1['bbox'][1] + obj1['bbox'][3]) / 2
                        center2_x = (obj2['bbox'][0] + obj2['bbox'][2]) / 2
                        center2_y = (obj2['bbox'][1] + obj2['bbox'][3]) / 2

                        dist = math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
                        
                        # Check if obj1 is behind obj2 (lower on screen) and too close, and obj1 is faster
                        if dist < TAILGATING_DISTANCE_THRESHOLD * self.global_ui_scale and \
                           obj1['velocity_y'] > obj2['velocity_y'] + TAILGATING_SPEED_DIFF_THRESHOLD and \
                           obj1['bbox'][3] > obj2['bbox'][3]: 
                            event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Driver Behavior: Tailgating (ID:{obj1['id']} behind ID:{obj2['id']})")
                            self.last_driver_behavior_alert_time = current_time
                            return # Only one driver behavior alert per cycle for simplicity
                    
                    # Aggressive Lane Change (simplified simulation)
                    if obj1['label'] in ["car", "truck", "bus"] and len(obj1.get('horizontal_movement_history', [])) == obj1['horizontal_movement_history'].maxlen:
                        avg_horizontal_movement = sum(obj1['horizontal_movement_history']) / len(obj1['horizontal_movement_history'])
                        if avg_horizontal_movement > AGGRESSIVE_LANE_CHANGE_THRESHOLD * self.global_ui_scale:
                            event_log_history.append(f"{datetime.now().strftime('%H:%M:%S')} - Driver Behavior: Aggressive Lane Change (ID:{obj1['id']})")
                            self.last_driver_behavior_alert_time = current_time
                            return # Only one driver behavior alert per cycle for simplicity

    def get_current_tracked_objects(self) -> List[Dict[str, Any]]:
        """Returns the currently tracked objects."""
        return list(self.tracked_objects.values())
    
    def get_all_detected_plates(self) -> deque:
        """Returns the deque of all detected unique plates."""
        return self.all_detected_plates

    def get_simulated_plate_details(self, plate_num: str) -> Dict[str, str]:
        """Provides simulated detailed info for a given plate number."""
        # This function simulates fetching more details from an "online database"
        # In a real system, this would be an API call.
        
        # Ensure the base data exists
        base_data = self._generate_simulated_plate_data(plate_num) 
        
        # Add more specific simulated details
        details = {**base_data} # Copy existing data
        
        vehicle_history = random.choice(["Clean", "Minor Incident Reported", "Major Accident Reported (Alert)"])
        insurance_status = random.choice(["Active", "Expired (Alert)", "Pending Renewal"])
        emission_status = random.choice(["Compliant", "Non-Compliant (Alert)"])
        last_service_date = (datetime.now() - timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d')
        
        details["Vehicle History"] = vehicle_history
        details["Insurance Status"] = insurance_status
        details["Emission Status"] = emission_status
        details["Last Service"] = last_service_date
        details["Traffic Violations"] = random.randint(0, 5)
        
        return details

# Import timedelta for simulated details
from datetime import timedelta
