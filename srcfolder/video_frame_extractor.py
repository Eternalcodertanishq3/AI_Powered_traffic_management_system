# srcfolder/utils/video_frame_extractor.py (New file)
import cv2
import os
import time

def extract_frames(video_path, output_folder, frames_per_second=1):
    """
    Extracts frames from a video file and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save the extracted frames.
        frames_per_second (int): How many frames to extract per second of video.
                                 Lower value = fewer frames, faster extraction, less redundancy.
    Returns:
        list: A list of paths to the extracted image files.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: Could not get FPS for {video_path}. Assuming 30 FPS.")
        fps = 30 # Default to 30 FPS if not detectable

    frame_interval = int(fps / frames_per_second) if frames_per_second > 0 else 1
    if frame_interval < 1: frame_interval = 1 # Ensure at least 1 frame per second if frames_per_second > fps

    extracted_paths = []
    frame_count = 0
    read_count = 0

    print(f"Extracting frames from '{video_path}' to '{output_folder}' (every {frame_interval} frames)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video or error

        if read_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_paths.append(frame_filename)
            frame_count += 1
        read_count += 1

    cap.release()
    print(f"Finished extracting {frame_count} frames from '{video_path}'.")
    return extracted_paths

if __name__ == "__main__":
    # Example usage:
    # Create a 'video_frames' folder in your 'data' directory
    video_path_1 = r"C:\Personal Projects\AI_Powered_traffic_management_system\data\gettyimages-2203212183-640_adpp.mp4"
    output_folder_1 = r"C:/Personal Projects/AI_Powered_traffic_management_system/data/video_frames/1643-148614430_frames"

    video_path_2 = r"C:\Personal Projects\AI_Powered_traffic_management_system\data\gettyimages-1936679257-640_adpp.mp4"
    output_folder_2 = r"C:/Personal Projects/AI_Powered_traffic_management_system/data/video_frames/getty_accident_frames"

    # It's better to save extracted frames to a structured folder.
    # You might want to label these extracted frames using your labeling_tool.py.

    extracted_frames_1 = extract_frames(video_path_1, output_folder_1, frames_per_second=1)
    extracted_frames_2 = extract_frames(video_path_2, output_folder_2, frames_per_second=1)

    print("Run this script to extract frames. Then load the output_folder into labeling_tool.py.")
    print("After labeling these frames, they will be included in your annotations.csv.")