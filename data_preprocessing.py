import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---

# 1. INPUT PATHS (Raw Data)
# Update these if your folder structure is slightly different
RAW_VIDEO_DIR = os.path.join('Suturing', 'Suturing', 'video')
RAW_KINEMATICS_DIR = os.path.join('Suturing', 'Suturing', 'kinematics')
RAW_TRANSCRIPTIONS_DIR = os.path.join('Suturing', 'Suturing', 'transcriptions')

# 2. EXPERIMENTAL SETUP PATHS (Splits)
# We look for the "SuperTrialOut" folder which contains splits like 1_Out, 2_Out, etc.
# If this path doesn't exist, the script falls back to looking for Train.txt/Test.txt in the root.
SPLIT_ROOT_DIR = os.path.join('Experimental_setup', 'Suturing', 'unBalanced', 'SkillDetection', 'SuperTrialOut')
FALLBACK_TRAIN = os.path.join('Experimental_setup', 'Suturing', 'unBalanced', 'Train.txt')
FALLBACK_TEST = os.path.join('Experimental_setup', 'Suturing', 'unBalanced', 'Test.txt')

# 3. OUTPUT DIRECTORY
OUTPUT_ROOT = 'data/jigsaws'
OUTPUT_IMAGES = os.path.join(OUTPUT_ROOT, 'images')
OUTPUT_KINEMATICS = os.path.join(OUTPUT_ROOT, 'kinematics')
OUTPUT_LABELS = os.path.join(OUTPUT_ROOT, 'labels')
OUTPUT_SPLITS = os.path.join(OUTPUT_ROOT, 'splits')

# 4. PROCESSING PARAMETERS
TARGET_FPS = 5
ORIGINAL_FPS = 30 
FRAME_SKIP = ORIGINAL_FPS // TARGET_FPS # 30 / 5 = 6
TARGET_WIDTH = 160

def parse_line(line):
    """Parses a single line of text into a clip dictionary."""
    parts = line.strip().split()
    if len(parts) < 2: return None
    
    filename_full = parts[0]
    try:
        score = float(parts[1])
    except ValueError:
        # If the second part isn't a number, it's likely a header or a Gesture Classification file
        return None

    # Filename format example: Suturing_C001_000026_003180.txt
    name_parts = filename_full.replace('.txt', '').split('_')
    
    # Reconstruct base video name (e.g., Suturing_C001)
    # The last two parts are start/end frames, everything before is the video name
    video_name = "_".join(name_parts[:-2]) 
    start_frame = int(name_parts[-2])
    end_frame = int(name_parts[-1])
    
    # Extract user ID (e.g., 'C' from 'C001')
    try:
        user_id = name_parts[1][0] 
    except IndexError:
        user_id = 'Unknown'

    return {
        'clip_name': filename_full.replace('.txt', ''),
        'video_name': video_name,
        'start': start_frame,
        'end': end_frame,
        'score': score,
        'user': user_id,
        'original_line': line.strip()
    }

def parse_split_file(file_path):
    """Reads a split file line by line."""
    if not os.path.exists(file_path): return []
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            res = parse_line(line)
            if res: data.append(res)
    return data

def find_all_splits():
    """Scans the Experimental_setup folder for all available splits."""
    splits = {} # format: {'split_name': {'train': [], 'test': []}}
    
    # 1. Search in SuperTrialOut (Priority)
    # Looking for structure: SuperTrialOut/1_Out/itr_1/train.txt
    if os.path.exists(SPLIT_ROOT_DIR):
        print(f"Scanning for splits in {SPLIT_ROOT_DIR}...")
        train_files = glob(os.path.join(SPLIT_ROOT_DIR, '**', 'train.txt'), recursive=True)
        
        for t_file in train_files:
            # Check for corresponding test file in the same directory
            parent = os.path.dirname(t_file)
            test_file = os.path.join(parent, 'test.txt')
            
            if os.path.exists(test_file):
                # Construct a readable split name (e.g., 1_Out_itr_1)
                rel_path = os.path.relpath(parent, SPLIT_ROOT_DIR)
                split_name = rel_path.replace(os.sep, '_')
                
                splits[split_name] = {
                    'train': parse_split_file(t_file),
                    'test': parse_split_file(test_file)
                }
    
    # 2. Fallback: Check for root level files if no experimental folder structure found
    if not splits:
        print("No nested splits found. Checking root/fallback files...")
        
        # Try finding Train.txt
        train_data = parse_split_file(FALLBACK_TRAIN)
        if not train_data: train_data = parse_split_file('Train.txt') # Try current dir
        
        # Try finding Test.txt
        test_data = parse_split_file(FALLBACK_TEST)
        if not test_data: test_data = parse_split_file('Test.txt') # Try current dir
        
        if train_data and test_data:
            splits['default_split'] = {'train': train_data, 'test': test_data}

    return splits

def process_clip(clip_info):
    """Extracts frames and kinematics for a single clip."""
    video_name = clip_info['video_name']
    clip_name = clip_info['clip_name']
    start_frame = clip_info['start']
    end_frame = clip_info['end']
    
    # --- 1. LOCATE VIDEO FILE ---
    # JIGSAWS videos often have two views: capture1 (left) and capture2 (right).
    # We prioritize capture1 as the primary view.
    possible_names = [
        f"{video_name}_capture1.avi",  # Priority 1: Left view
        f"{video_name}_capture2.avi",  # Priority 2: Right view
        f"{video_name}.avi",            # Priority 3: Base name
        f"{video_name}_capture1.mp4",
        f"{video_name}_capture2.mp4",
        f"{video_name}.mp4"
    ]
    
    vid_path = None
    # Check exact paths first inside the RAW_VIDEO_DIR
    for name in possible_names:
        p = os.path.join(RAW_VIDEO_DIR, name)
        if os.path.exists(p):
            vid_path = p
            break
            
    # Fallback: Recursive search if not found in top dir (e.g., subfolders)
    if not vid_path:
        # Try finding capture1 recursively
        found = glob(os.path.join(RAW_VIDEO_DIR, '**', f"{video_name}*capture1.avi"), recursive=True)
        if found:
            vid_path = found[0]
        else:
            # Try finding base name recursively
            found = glob(os.path.join(RAW_VIDEO_DIR, '**', f"{video_name}.avi"), recursive=True)
            if found:
                vid_path = found[0]
            else:
                # Last resort: Try finding ANY file starting with the video name
                found = glob(os.path.join(RAW_VIDEO_DIR, '**', f"{video_name}*.avi"), recursive=True)
                if found:
                    vid_path = found[0]
                else:
                    print(f"Skipping {clip_name}: Video not found for {video_name}")
                    return None

    # --- 2. LOCATE KINEMATICS FILE ---
    kin_path = os.path.join(RAW_KINEMATICS_DIR, video_name + '.txt')
    if not os.path.exists(kin_path):
         # Recursive search
         found = glob(os.path.join(RAW_KINEMATICS_DIR, '**', video_name + '.txt'), recursive=True)
         if found: 
             kin_path = found[0]
         else: 
             print(f"Skipping {clip_name}: Kinematics not found for {video_name}")
             return None

    # Setup output paths for this specific clip
    save_img_dir = os.path.join(OUTPUT_IMAGES, clip_name)
    
    # Optimization: Skip if already processed (check if folder exists and has images)
    if os.path.exists(save_img_dir) and len(os.listdir(save_img_dir)) > 0:
        return clip_name, clip_info['score']
    
    os.makedirs(save_img_dir, exist_ok=True)
    
    # --- 3. PROCESS VIDEO FRAMES ---
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Safety check for frame range
    if end_frame > total_frames: end_frame = total_frames

    processed_indices = []
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Logic: Only save frames within the clip range
        if frame_idx >= start_frame and frame_idx <= end_frame:
            # Downsample logic: (frame_idx - start) % SKIP == 0
            if (frame_idx - start_frame) % FRAME_SKIP == 0:
                # Resize to target width (keeping aspect ratio)
                h, w, _ = frame.shape
                aspect_ratio = h / w
                new_h = int(TARGET_WIDTH * aspect_ratio)
                resized_frame = cv2.resize(frame, (TARGET_WIDTH, new_h))
                
                # Save as 00000.jpg, 00001.jpg, etc.
                fname = f"{saved_count:05d}.jpg"
                cv2.imwrite(os.path.join(save_img_dir, fname), resized_frame)
                
                processed_indices.append(frame_idx)
                saved_count += 1
        
        frame_idx += 1
        if frame_idx > end_frame: break
            
    cap.release()

    # --- 4. PROCESS KINEMATICS ---
    try:
        kin_data = np.loadtxt(kin_path)
        
        # Match kinematics rows to the EXACT video frames we saved
        # We use min() to clamp indices in case kinematics is slightly shorter than video
        valid_indices = [min(i, len(kin_data)-1) for i in processed_indices]
        
        clip_kinematics = kin_data[valid_indices]
        
        # Save as .npy
        np.save(os.path.join(OUTPUT_KINEMATICS, f"{clip_name}.npy"), clip_kinematics)
        
    except Exception as e:
        print(f"Error processing kinematics for {clip_name}: {e}")

    return clip_name, clip_info['score']

def main():
    print("========================================")
    print("   JIGSAWS Data Preprocessing for ViSA  ")
    print("========================================")
    
    # Create output directories
    for p in [OUTPUT_IMAGES, OUTPUT_KINEMATICS, OUTPUT_LABELS, OUTPUT_SPLITS]:
        os.makedirs(p, exist_ok=True)

    # 1. Discover Splits
    splits = find_all_splits()
    if not splits:
        print("ERROR: No valid split files (Train.txt/Test.txt) found.")
        print(f"Checked: {SPLIT_ROOT_DIR}")
        print("Please check your 'Experimental_setup' folder or put Train.txt in the root.")
        return
    
    print(f"Found {len(splits)} split(s): {list(splits.keys())}")

    # 2. Collect ALL unique clips from ALL splits
    # This prevents processing the same video twice if it appears in multiple splits
    unique_clips = {}
    for split_name, data in splits.items():
        for clip in data['train'] + data['test']:
            unique_clips[clip['clip_name']] = clip
            
    print(f"Total unique clips to process: {len(unique_clips)}")

    # 3. Process Clips (Extract Images/Kinematics)
    results = []
    print("Processing clips...")
    for clip_name, clip_data in tqdm(unique_clips.items()):
        res = process_clip(clip_data)
        if res:
            results.append(res) # Format: (ClipName, Score)

    if not results:
        print("No clips were successfully processed. Check video paths.")
        return

    # 4. Save Master Labels CSV
    df_all = pd.DataFrame(results, columns=['ClipName', 'GRS_Score'])
    df_all.to_csv(os.path.join(OUTPUT_LABELS, 'all_labels.csv'), index=False)

    # 5. Generate CSVs for EACH split found
    print("\nGenerating Split CSVs...")
    for split_name, data in splits.items():
        # Create a folder for this split (e.g., data/jigsaws/splits/1_Out_itr_1)
        split_dir = os.path.join(OUTPUT_SPLITS, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Get list of clip names for this split's train and test
        train_names = {c['clip_name'] for c in data['train']}
        test_names = {c['clip_name'] for c in data['test']}
        
        # Filter the master results to create the CSVs
        df_train = df_all[df_all['ClipName'].isin(train_names)]
        df_test = df_all[df_all['ClipName'].isin(test_names)]
        
        df_train.to_csv(os.path.join(split_dir, 'train_labels.csv'), index=False)
        df_test.to_csv(os.path.join(split_dir, 'test_labels.csv'), index=False)
        
        # If this is the "default" or first split, also save to main labels folder for easy loading
        if split_name == list(splits.keys())[0]:
            df_train.to_csv(os.path.join(OUTPUT_LABELS, 'train_labels.csv'), index=False)
            df_test.to_csv(os.path.join(OUTPUT_LABELS, 'test_labels.csv'), index=False)
            print(f" -> Set '{split_name}' as the default active split in {OUTPUT_LABELS}")

    print("\n========================================")
    print("Processing Complete!")
    print(f"Data saved to: {os.path.abspath(OUTPUT_ROOT)}")
    print("========================================")

if __name__ == "__main__":
    main()