# srcfolder/person_tracker.py

import numpy as np

class SimplePersonTracker:
    """
    A basic person tracker that assigns unique IDs and estimates counts.
    It's based on IoU (Intersection over Union) for matching and has simple
    logic for adding new tracks and ending old ones.
    For robust, long-term tracking, consider more advanced algorithms (e.g., DeepSORT).
    """
    def __init__(self, iou_threshold=0.3, max_unmatched_frames=10):
        self.next_id = 0
        self.tracked_persons = {} # {id: {'bbox': [x1,y1,x2,y2], 'frames_unmatched': 0, 'total_detections': 1}}
        self.iou_threshold = iou_threshold
        self.max_unmatched_frames = max_unmatched_frames
        self.total_unique_persons_completed = 0 # Count of persons whose tracks have been officially terminated

    def _iou(self, box1, box2):
        """Calculates Intersection over Union (IoU) of two bounding boxes."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        intersection_area = inter_width * inter_height

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) * 1.0 # Ensure float division
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]) * 1.0
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0
        return intersection_area / union_area

    def update(self, current_person_boxes):
        """
        Updates tracked persons based on current frame's detections.

        Args:
            current_person_boxes (np.array): A NumPy array of bounding boxes for persons
                                             in the current frame, format [x1, y1, x2, y2].

        Returns:
            list: A list of (person_id, bbox) tuples for currently tracked persons.
        """
        matched_detections = [False] * len(current_person_boxes)
        updated_tracked_persons = {}

        # 1. Try to match current detections with existing tracks
        for person_id, track_info in self.tracked_persons.items():
            best_iou = -1
            best_det_idx = -1

            for det_idx, det_box in enumerate(current_person_boxes):
                if not matched_detections[det_idx]:
                    iou = self._iou(track_info['bbox'], det_box)
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_det_idx = det_idx
            
            if best_det_idx != -1:
                # Match found: update track's bbox and reset unmatched frames counter
                updated_tracked_persons[person_id] = {
                    'bbox': current_person_boxes[best_det_idx],
                    'frames_unmatched': 0,
                    'total_detections': track_info['total_detections'] + 1
                }
                matched_detections[best_det_idx] = True
            else:
                # No match found: increment unmatched frames counter
                track_info['frames_unmatched'] += 1
                if track_info['frames_unmatched'] <= self.max_unmatched_frames:
                    # Keep track if within grace period
                    updated_tracked_persons[person_id] = track_info
                else:
                    # Track lost (person left frame or was not detected for too long)
                    self.total_unique_persons_completed += 1
                    # print(f"Person ID {person_id} lost. Total completed: {self.total_unique_persons_completed}")


        # 2. Add new detections as new tracks
        for det_idx, det_box in enumerate(current_person_boxes):
            if not matched_detections[det_idx]:
                self.next_id += 1
                updated_tracked_persons[self.next_id] = {
                    'bbox': det_box,
                    'frames_unmatched': 0,
                    'total_detections': 1
                }
        
        self.tracked_persons = updated_tracked_persons

        # Return list of current (id, bbox) tuples (can be used for drawing tracked IDs)
        return [(p_id, info['bbox']) for p_id, info in self.tracked_persons.items()]

    def get_total_unique_persons(self):
        """
        Returns the total count of unique persons detected and either completed their track
        or are still actively tracked at the moment of query. This is a proxy for "total crossed/went".
        """
        return self.total_unique_persons_completed + len(self.tracked_persons)

    def reset(self):
        """Resets the tracker for a new session/video."""
        self.next_id = 0
        self.tracked_persons = {}
        self.total_unique_persons_completed = 0

# Example Usage (for testing this tracker directly):
if __name__ == "__main__":
    tracker = SimplePersonTracker()

    # Simulate frames with detections
    print("Frame 1:")
    frame1_detections = np.array([[10,10,20,20], [30,30,40,40]])
    tracked = tracker.update(frame1_detections)
    print(f"Tracked: {tracked}")
    print(f"Unique persons counted so far (active + lost): {tracker.get_total_unique_persons()}")

    print("\nFrame 2:")
    frame2_detections = np.array([[11,11,21,21], [50,50,60,60]]) # Person 1 moved, Person 2 disappeared, New Person 3
    tracked = tracker.update(frame2_detections)
    print(f"Tracked: {tracked}")
    print(f"Unique persons counted so far (active + lost): {tracker.get_total_unique_persons()}")


    print("\nFrame 3 (after max_unmatched_frames, Person 2 should be counted as lost):")
    frame3_detections = np.array([[12,12,22,22], [51,51,61,61]]) # Person 1 and 3 still there
    tracked = tracker.update(frame3_detections)
    print(f"Tracked: {tracked}")
    print(f"Unique persons counted so far (active + lost): {tracker.get_total_unique_persons()}")


    print("\nEnd of video (simulated).")
    # At the end of a video, the get_total_unique_persons() naturally includes remaining active tracks.
    print(f"Final total unique persons counted: {tracker.get_total_unique_persons()}")

    print("\nResetting tracker for a new video.")
    tracker.reset()
    print(f"Total unique persons after reset: {tracker.get_total_unique_persons()}")

