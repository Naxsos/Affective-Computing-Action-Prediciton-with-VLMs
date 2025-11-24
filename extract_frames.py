#!/usr/bin/env python3
"""
Extract n frames before a specific timestamp from a video file.
"""

import cv2
import argparse
import os
from pathlib import Path
from datetime import datetime


def extract_frames_before_timestamp(video_path, timestamp_seconds, num_frames, output_dir="output_frames", frame_interval=1, auto_subfolder=True):
    """
    Extract n frames before a specific timestamp from a video.
    
    Args:
        video_path (str): Path to the video file
        timestamp_seconds (float): Timestamp in seconds where to extract frames before
        num_frames (int): Number of frames to extract before the timestamp
        output_dir (str): Base directory to save the extracted frames
        frame_interval (int): Interval between frames (e.g., 5 means every 5th frame going backwards)
        auto_subfolder (bool): If True, creates a timestamped subfolder for each extraction
    
    Returns:
        list: Paths to the saved frames
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video Properties:")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Target Timestamp: {timestamp_seconds} seconds")
    
    # Validate timestamp
    if timestamp_seconds > duration:
        raise ValueError(f"Timestamp {timestamp_seconds}s exceeds video duration {duration:.2f}s")
    
    if timestamp_seconds < 0:
        raise ValueError("Timestamp must be positive")
    
    # Calculate the target frame number
    target_frame = int(timestamp_seconds * fps)
    
    # Calculate frame indices going backwards with interval
    frame_indices = []
    for i in range(num_frames):
        frame_idx = target_frame - (i * frame_interval)
        if frame_idx >= 0:
            frame_indices.append(frame_idx)
        else:
            break
    
    # Reverse to go from oldest to newest
    frame_indices.reverse()
    
    actual_num_frames = len(frame_indices)
    
    if actual_num_frames < num_frames:
        print(f"Warning: Only {actual_num_frames} frames available before timestamp with interval {frame_interval}")
    
    if frame_indices:
        print(f"\nExtracting {actual_num_frames} frames from {frame_indices[0]} to {frame_indices[-1]} (interval: {frame_interval})")
    
    # Create output directory with optional timestamped subfolder
    if auto_subfolder:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        subfolder_name = f"{video_name}_t{timestamp_seconds}s_n{num_frames}_i{frame_interval}_{timestamp_str}"
        output_path = Path(output_dir) / subfolder_name
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_frames = []
    
    # Extract frames at specified indices
    for frame_number in frame_indices:
        # Set video to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_number}")
            continue
        
        # Generate filename with frame number and timestamp
        frame_timestamp = frame_number / fps
        filename = f"frame_{frame_number:06d}_time_{frame_timestamp:.3f}s.jpg"
        filepath = output_path / filename
        
        # Save frame
        cv2.imwrite(str(filepath), frame)
        saved_frames.append(str(filepath))
        
        print(f"  Saved: {filename}")
    
    cap.release()
    
    print(f"\nSuccessfully extracted {len(saved_frames)} frames to '{output_path}/'")
    
    return saved_frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract n frames before a specific timestamp from a video file."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "timestamp",
        type=float,
        help="Timestamp in seconds (e.g., 10.5 for 10.5 seconds)"
    )
    parser.add_argument(
        "num_frames",
        type=int,
        help="Number of frames to extract before the timestamp"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_frames",
        help="Output directory for extracted frames (default: output_frames)"
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=1,
        help="Frame interval - extract every Nth frame going backwards (default: 1, consecutive frames)"
    )
    parser.add_argument(
        "--no-subfolder",
        action="store_true",
        help="Disable automatic timestamped subfolder creation (will overwrite existing files)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    if args.num_frames <= 0:
        print("Error: Number of frames must be positive")
        return 1
    
    try:
        extract_frames_before_timestamp(
            args.video_path,
            args.timestamp,
            args.num_frames,
            args.output,
            args.interval,
            auto_subfolder=not args.no_subfolder
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

