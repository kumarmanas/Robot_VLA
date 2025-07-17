"""
Utility functions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
def setup_logging(log_level: str = "INFO"):
    """lof config"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('wheel_loader_vla.log'),
            logging.StreamHandler()
        ]
    )

def save_json(data: Any, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved data to {filepath}")

def load_json(filepath: str) -> Any:
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded data from {filepath}")
    return data

def analyze_dataset_statistics(data_dir: str):
    vqa_file = Path(data_dir) / "vqa_pairs.json"
    if not vqa_file.exists():
        logger.error(f"VQA file not found: {vqa_file}")
        return
    vqa_pairs = load_json(vqa_file)
    total_pairs = len(vqa_pairs)
    unique_questions = len(set(pair["question"] for pair in vqa_pairs))
    unique_actions = len(set(pair["answer"] for pair in vqa_pairs))
    action_types = {}
    for pair in vqa_pairs:
        action = pair["answer"]
        action_type = action.split("(")[0] if "(" in action else action
        action_types[action_type] = action_types.get(action_type, 0) + 1
    
    logger.info("Dataset Stats")
    logger.info(f"Total VQA pairs: {total_pairs}")
    logger.info(f"Unique questions: {unique_questions}")
    logger.info(f"Unique actions: {unique_actions}")
    logger.info(f"Action type distribution: {action_types}")