"""
Batch Hand Landmark Data Collection
Part of MS-SLR (Motion-Signature Sign Language Recognition)

Author: Mohammed Hashir Firoze
Institution: University of Wollongong in Dubai
Date: February 2026

Description:
    Processes all videos in a folder and extracts MediaPipe hand landmarks
    for each frame, generating a dataset CSV for model training.
    
    Key features:
    - Two-hand tracking: Captures both hands simultaneously
    - Consistent hand ordering: Always hand_a (left) before hand_b (right)
    - 126 features per frame: 21 landmarks × 3 coordinates × 2 hands
    - WLASL video ID mapping: Automatic sign label resolution
    
    Output format:
    label, video_id, frame_index, x0, y0, z0, ..., x20_b, y20_b, z20_b

Usage:
    # Process entire WLASL dataset
    python batch_collect_two_hand_data.py --videos-dir "C:/SignLanguageProject/videos" --output hand_data_2hands.csv
    
    # Process every 2nd frame (faster, still accurate)
    python batch_collect_two_hand_data.py --videos-dir "videos/" --frame-step 2
    
    # Limit to 50 frames per video (for testing)
    python batch_collect_two_hand_data.py --videos-dir "videos/" --max-frames 50

Performance:
    - ~15-30 videos/minute on typical CPU
    - 500 videos = ~20-35 minutes
    - Memory: <2GB RAM
"""

import argparse
import csv
import json
import os
import sys
import cv2
import mediapipe as mp
import time


def build_video_to_gloss(wlasl_json_path):
    """
    Parse WLASL JSON and create video_id → sign gloss mapping.
    
    WLASL videos are named by video_id (e.g., "12345.mp4"), not sign names.
    This function creates a lookup table so we can label our dataset properly.
    
    Handles multiple ID formats:
    - Raw video_id: "12345"
    - Integer format: 12345
    - With extension: "12345.mp4"
    
    Returns:
        dict: video_id → sign_name mapping
    """
    with open(wlasl_json_path, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)

    video_to_gloss = {}
    for entry in wlasl_data:
        gloss = entry.get("gloss", "")
        for instance in entry.get("instances", []):
            video_id = instance.get("video_id", "")
            if not video_id:
                continue
            
            # Store all possible formats of this ID
            video_to_gloss[video_id] = gloss
            
            # Integer version (some datasets use int IDs)
            try:
                video_id_int = str(int(video_id))
                video_to_gloss[video_id_int] = gloss
            except ValueError:
                pass
            
            # With .mp4 extension (common in file listings)
            video_to_gloss[video_id + ".mp4"] = gloss
            try:
                video_to_gloss[str(int(video_id)) + ".mp4"] = gloss
            except ValueError:
                pass
    
    return video_to_gloss


def smart_lookup(label, video_to_gloss):
    """
    Robust label matching for various video naming conventions.
    
    Input videos might be named:
    - "12345"         → video ID
    - "12345.mp4"     → video ID with extension
    - "help"          → direct sign name (rare)
    
    Returns sign name if found, None otherwise.
    """
    label = str(label).strip()
    
    # Direct match
    if label in video_to_gloss:
        return video_to_gloss[label]
    
    # Try without extension
    label_no_ext = label.replace(".mp4", "").replace(".MP4", "")
    if label_no_ext in video_to_gloss:
        return video_to_gloss[label_no_ext]
    
    # Try as integer
    try:
        label_int = str(int(label_no_ext))
        if label_int in video_to_gloss:
            return video_to_gloss[label_int]
    except ValueError:
        pass
    
    return None


def extract_two_hand_features(results):
    """
    Extract and order hand landmarks from MediaPipe detection.
    
    Critical design decision: Consistent hand ordering
    - Always place left hand (camera POV) in hand_a
    - Right hand (camera POV) in hand_b
    - Use wrist X-coordinate for left/right determination
    
    Why this matters: Model needs consistent feature ordering.
    Without this: "HELLO" signed with right hand looks different from
    left hand in the feature space → 2x vocabulary size needed.
    
    Returns:
        list: 126 floats (hand_a: x0..z20, hand_b: x0_b..z20_b)
              If one hand detected: hand_b filled with zeros
              If no hands: all zeros (filtered later in training)
    """
    hand_a = [0.0] * 63  # 21 landmarks × 3 coords
    hand_b = [0.0] * 63
    hands_data = []

    # Extract all detected hands
    for hand_landmarks in results.multi_hand_landmarks:
        coords = []
        for landmark in hand_landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])
        hands_data.append(coords)

    # Assign to hand_a and hand_b based on position
    if len(hands_data) == 1:
        hand_a = hands_data[0]  # Single hand → use as hand_a
    elif len(hands_data) >= 2:
        # Sort by wrist X-coordinate (landmark 0)
        # Lower X = left side = hand_a
        wrist_x = [
            hand.landmark[0].x for hand in results.multi_hand_landmarks[:2]
        ]
        first_idx = 0 if wrist_x[0] <= wrist_x[1] else 1
        second_idx = 1 - first_idx
        hand_a = hands_data[first_idx]
        hand_b = hands_data[second_idx]

    return hand_a + hand_b


def main():
    parser = argparse.ArgumentParser(
        description="Batch collect two-hand landmark data from video folder"
    )
    parser.add_argument(
        "--videos-dir", 
        required=True, 
        help="Path to folder containing sign language videos"
    )
    parser.add_argument(
        "--output", 
        default="hand_data_2hands.csv", 
        help="Output CSV file path (default: hand_data_2hands.csv)"
    )
    parser.add_argument(
        "--wlasl", 
        default="WLASL_v0.3.json", 
        help="WLASL JSON file for video-to-sign mapping (default: WLASL_v0.3.json)"
    )
    parser.add_argument(
        "--frame-step", 
        type=int, 
        default=1, 
        help="Process every Nth frame (1=all frames, 2=every other, etc.)"
    )
    parser.add_argument(
        "--max-frames", 
        type=int, 
        default=0, 
        help="Max frames per video (0=unlimited, 50=good for testing)"
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.videos_dir):
        print(f"ERROR: Videos directory not found: {args.videos_dir}")
        print("Please provide valid --videos-dir path")
        sys.exit(1)
    
    if not os.path.exists(args.wlasl):
        print(f"ERROR: WLASL JSON not found: {args.wlasl}")
        print("Download from: https://github.com/dxli94/WLASL")
        sys.exit(1)

    # Build video ID → sign name mapping
    print("Loading WLASL dictionary...")
    video_to_gloss = build_video_to_gloss(args.wlasl)
    print(f"✓ Mapped {len(video_to_gloss)} video IDs to signs")

    # Prepare CSV headers
    # Format: label, video_id, frame_index, x0..z20 (hand_a), x0_b..z20_b (hand_b)
    headers = ["label", "video_id", "frame_index"]
    for i in range(21):
        headers.extend([f"x{i}", f"y{i}", f"z{i}"])
    for i in range(21):
        headers.extend([f"x{i}_b", f"y{i}_b", f"z{i}_b"])

    # Create or append to CSV
    file_exists = os.path.exists(args.output)
    if not file_exists:
        with open(args.output, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Created new CSV: {args.output}")
    else:
        print(f"Appending to existing CSV: {args.output}")

    csv_file = open(args.output, mode="a", newline="")
    csv_writer = csv.writer(csv_file)

    # Initialize MediaPipe Hands
    # Using 2-hand mode with moderate confidence for good detection/speed balance
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,      # Video mode (faster, uses temporal tracking)
        max_num_hands=2,               # Detect both hands
        min_detection_confidence=0.5,  # Standard threshold (0.5 = 50%)
        min_tracking_confidence=0.5,   # Maintain tracking across frames
    )

    # Find all video files
    video_files = [
        f for f in os.listdir(args.videos_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ]
    total_videos = len(video_files)
    
    if total_videos == 0:
        print(f"ERROR: No video files found in {args.videos_dir}")
        print("Supported formats: .mp4, .mov, .avi, .mkv")
        sys.exit(1)

    print(f"\nFound {total_videos} videos. Starting batch processing...")
    print(f"Frame step: {args.frame_step} (processing every {args.frame_step} frame(s))")
    if args.max_frames > 0:
        print(f"Max frames per video: {args.max_frames}")
    print("-" * 80)

    total_rows = 0
    start_time = time.time()

    # Process each video
    for idx, filename in enumerate(video_files, start=1):
        path = os.path.join(args.videos_dir, filename)
        
        # Resolve video → sign name
        label = smart_lookup(filename, video_to_gloss)
        if not label:
            # Skip videos not in WLASL (corrupted/extra files)
            continue
        
        # Extract video ID (used for temporal grouping in training)
        video_id = os.path.splitext(os.path.basename(filename))[0]
        
        # Open video
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"⚠ Skipped {filename} (could not open)")
            continue
        
        frame_index = 0
        saved = 0
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            frame_index += 1
            
            # Skip frames if frame_step > 1
            if args.frame_step > 1 and (frame_index % args.frame_step) != 0:
                continue
            
            # Convert BGR → RGB (MediaPipe expects RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False  # Performance optimization
            
            # Detect hands
            results = hands.process(rgb)
            rgb.flags.writeable = True
            
            # Save if hands detected
            if results.multi_hand_landmarks:
                row = [label, video_id, frame_index]
                row.extend(extract_two_hand_features(results))
                csv_writer.writerow(row)
                total_rows += 1
                saved += 1
            
            # Stop if max frames reached
            if args.max_frames and saved >= args.max_frames:
                break
        
        cap.release()
        
        # Progress reporting
        elapsed = time.time() - start_time
        avg_per_video = elapsed / idx if idx else 0
        remaining = (total_videos - idx) * avg_per_video
        percent = (idx / total_videos * 100) if total_videos else 100
        
        print(
            f"[{idx:3d}/{total_videos}] {percent:5.1f}% | "
            f"Rows: {total_rows:6d} | {filename:40s} | "
            f"ETA: {remaining/60:4.1f} min"
        )

    # Cleanup
    hands.close()
    csv_file.close()
    
    elapsed_total = time.time() - start_time
    print("-" * 80)
    print(f"\n✓ Collection complete!")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Output: {args.output}")
    print(f"  Time: {elapsed_total/60:.1f} minutes")
    print(f"  Rate: {total_videos/(elapsed_total/60):.1f} videos/minute")


if __name__ == "__main__":
    main()
