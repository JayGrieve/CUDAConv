#!/usr/bin/env python3
import cv2
import sys
import os
import glob
import re

def natural_sort_key(s):
    """Sort strings containing numbers in natural order"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_video(input_dir="output_images", output_video="output.mp4", fps=30):
    # Get all PNG files from the directory
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not image_files:
        print(f"Error: No PNG files found in {input_dir}")
        return
    
    # Sort files naturally (0.png, 1.png, 2.png, ..., 10.png, 11.png, etc.)
    image_files.sort(key=natural_sort_key)
    
    print(f"Found {len(image_files)} images")
    print(f"First image: {image_files[0]}")
    print(f"Last image: {image_files[-1]}")
    
    # Read the first image to get dimensions
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Error: Could not read first image: {image_files[0]}")
        return
    
    height, width, channels = first_frame.shape
    print(f"Video dimensions: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID', 'H264', etc.
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open video writer")
        return
    
    # Write each frame to the video
    for i, image_file in enumerate(image_files):
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: Could not read {image_file}, skipping...")
            continue
        
        out.write(frame)
        
        if i % 100 == 0:
            print(f"Processed frame {i}/{len(image_files)}")
    
    # Release everything
    out.release()
    print(f"\nVideo created successfully: {output_video}")
    print(f"Total frames: {len(image_files)}")

if __name__ == "__main__":
    input_dir = "output_images"
    output_video = "output.mp4"
    fps = 30
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_video = sys.argv[2]
    if len(sys.argv) > 3:
        fps = int(sys.argv[3])
    
    print(f"Input directory: {input_dir}")
    print(f"Output video: {output_video}")
    print(f"FPS: {fps}\n")
    
    create_video(input_dir, output_video, fps)
