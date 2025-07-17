#!/usr/bin/env python3
"""
Vision Language Action Model, main script landing page
"""
import argparse
import logging
import json
from pathlib import Path
import sys
from data_preparation import prepare_dataset
from vqa_generation import generate_vqa_pairs
from object_detection import ObjectDetector
from model_training import train_action_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="VLA Model")
    parser.add_argument("--mode", choices=["prepare", "train", "generate","gradio"], required=True,
                       help="Mode to run: data prep, generate VQA, model train, run demo")
    parser.add_argument("--data_dir", default="./data", help="Data dir")
    parser.add_argument("--model_dir", default="./models", help="Model dir")
    parser.add_argument("--video_urls", default="video_urls.txt", help="YouTube URLs")
    parser.add_argument("--max_videos", type=int, default=5, help="Max videos to process")
    parser.add_argument("--frames_per_video", type=int, default=50, help="Frames per video")
    parser.add_argument("--use_detection", action="store_true", default=True,
                       help="Use object detection for VQA")
    parser.add_argument("--confidence_threshold", type=float, default=0.4,
                       help="Confidence threshold for object detection")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--model_name", default="unsloth/Llama-3.2-1B-Instruct",
                       help="Base model name for fine-tuning")
    
    args = parser.parse_args()
    Path(args.data_dir).mkdir(exist_ok=True)
    Path(args.model_dir).mkdir(exist_ok=True)
    Path(args.model_dir, "checkpoints").mkdir(exist_ok=True)
    Path(args.model_dir, "final_model").mkdir(exist_ok=True)
    try:
        if args.mode == "prepare":
            # Step 1: data prep to download videos and extract frames
            prepare_dataset(args.video_urls, args.data_dir, args.max_videos, args.frames_per_video)
            # Step 2: Validate frame extraction
            frame_paths_file = Path(args.data_dir) / "frame_paths.txt"
            if frame_paths_file.exists():
                with open(frame_paths_file, 'r') as f:
                    frame_count = len(f.readlines())
            else:
                logger.error("Frame extraction failed")
                sys.exit(1)
            # Step 3: automatic generation of VQA pairs with object detection as asked in task
            logger.info("Generating VQA pairs with object detection")
            generate_vqa_pairs(args.data_dir, use_detection=args.use_detection, 
                             confidence_threshold=args.confidence_threshold)
            # Step 4: validate VQA generation
            vqa_file = Path(args.data_dir) / "vqa_pairs.json"
            if vqa_file.exists():
                with open(vqa_file, 'r') as f:
                    vqa_data = json.load(f)
                single_actions = sum(1 for item in vqa_data if item.get("action_type") == "single")
                sequences = sum(1 for item in vqa_data if item.get("action_type") == "sequence")
                logger.info(f"Detection usage: {'Enabled' if args.use_detection else 'Disabled'}")
        # To generate VQA Pairs Only    
        elif args.mode == "generate":
            frame_paths_file = Path(args.data_dir) / "frame_paths.txt"
            if not frame_paths_file.exists():
                logger.error("No frame_paths.txt found")
                sys.exit(1)
            generate_vqa_pairs(args.data_dir, use_detection=args.use_detection,
                             confidence_threshold=args.confidence_threshold)
            
        elif args.mode == "gradio":
            logger.info("**Gradio Web demo Interface**")
            final_model_dir = Path(args.model_dir) / "final_model"
            if not final_model_dir.exists():
                logger.error("No trained model found. Run 'train' mode first.")
                sys.exit(1)
            from gradio_demo import create_simple_demo
            demo = create_simple_demo()
            demo.launch(share=True, inline=True)

        # VLA training
        elif args.mode == "train": 
            vqa_file = Path(args.data_dir) / "vqa_pairs.json"
            if not vqa_file.exists():
                logger.error("No VQA data found")
                sys.exit(1)
            train_action_model(
                data_dir=args.data_dir,
                model_dir=args.model_dir,
                base_model=args.model_name,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs
            )
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()