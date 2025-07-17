"""
Data prep module for downloading YouTube videos and extracting frames
"""

import os
import cv2
import yt_dlp
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def download_video(url: str, output_dir: str) -> str:
    """Download video from YouTube URL"""
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return None

def extract_frames(video_path: str, output_dir: str, max_frames: int = 50) -> List[str]:
    """Extract frames from video at regular intervals"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #frame interval count
    interval = max(1, total_frames // max_frames)
    frame_paths = []
    frame_count = 0
    video_name = Path(video_path).stem
    frame_dir = Path(output_dir) / "frames" / video_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_filename = frame_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            frame_paths.append(str(frame_filename))
        frame_count += 1
        if len(frame_paths) >= max_frames:
            break
    cap.release()
    logger.info(f"Extracted {len(frame_paths)} frames from {video_path}")
    return frame_paths

def prepare_dataset(urls_file: str, data_dir: str, max_videos: int, frames_per_video: int):
    """prepare dataset from YouTube videos"""
    # Default URLs if file doesn't exist
    default_urls = [
        "https://www.youtube.com/watch?v=o5LxOWSQSIk",
        "https://www.youtube.com/watch?v=hWp9vZ7eeaM", 
        "https://www.youtube.com/watch?v=PnFhMHbcL44",
        "https://www.youtube.com/watch?v=s7K23lRaLwA",
        "https://www.youtube.com/watch?v=wc72kf9DWaY"
    ]
    try:
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        logger.warning(f"URLs file {urls_file} not found, using default URLs")
        urls = default_urls
        with open(urls_file, 'w') as f:
            f.write('\n'.join(default_urls))
    urls = urls[:max_videos]
    video_dir = Path(data_dir) / "videos"
    video_dir.mkdir(exist_ok=True)  
    all_frame_paths = []
    
    for i, url in enumerate(urls):
        logger.info(f"Processing video {i+1}/{len(urls)}: {url}")
        video_path = download_video(url, str(video_dir))
        if not video_path:
            continue
        frame_paths = extract_frames(video_path, data_dir, frames_per_video)
        all_frame_paths.extend(frame_paths)
        try:
            os.remove(video_path)
        except:
            pass
    
    logger.info(f"Total frames extracted: {len(all_frame_paths)}")
    # Save frame paths
    with open(Path(data_dir) / "frame_paths.txt", 'w') as f:
        f.write('\n'.join(all_frame_paths))