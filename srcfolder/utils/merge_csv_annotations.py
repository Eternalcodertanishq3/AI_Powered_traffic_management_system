# srcfolder/utils/merge_csv_annotations.py

import pandas as pd
import os

def merge_csvs(input_csv_paths, output_csv_path="combined_annotations.csv"):
    all_dfs = []
    for path in input_csv_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                all_dfs.append(df)
                print(f"Loaded {len(df)} entries from {os.path.basename(path)}")
            except Exception as e:
                print(f"[ERROR] Could not read CSV file {path}: {e}")
        else:
            print(f"[WARNING] CSV file not found: {path}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.drop_duplicates(subset=['image_path'], inplace=True)
        combined_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully merged {len(combined_df)} unique entries into {output_csv_path}")
    else:
        print("No CSV files found to merge. 'combined_annotations.csv' will not be created.")

if __name__ == "__main__":
    # >>> CRITICAL: List ALL your annotation CSV files here. <<<
    csv_files_to_merge = [
        "annotations.csv",                   # Your main project annotations
        "traf_acci_data_annotations.csv",    # <--- Corrected name here!
        # Add any other CSVs from video frame extractions if you have them:
        # "my_video_frames_annotations.csv",
    ]

    merge_csvs(csv_files_to_merge, "combined_annotations.csv")
    print("\nReady to train! Now update 'annotations_csv_path' in train_scene_classifier.py to 'combined_annotations.csv'.")