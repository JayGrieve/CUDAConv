#!/usr/bin/env python3
import cv2
import sys
import os

def extract_frames(video_path, output_dir="frames"):
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: File does not exist: {video_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    print(f"Video opened successfully!")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total frames: {frame_count}")
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame as i.png
        filename = os.path.join(output_dir, f"{frame_idx}.png")
        cv2.imwrite(filename, frame)
        
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}")
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {frame_idx} frames into '{output_dir}' directory.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py <video.mp4>")
        sys.exit(1)
    
    extract_frames(sys.argv[1])
