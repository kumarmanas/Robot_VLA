"""
VQA pair generation with obj. detection knowledge
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

logger = logging.getLogger(__name__)
class VQAGenerator:
    def __init__(self):
        """Initialize the VQA generator with BLIP model"""
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # Action templates for wheel loader ops
        self.action_templates = [
            "drive_to_pile({x}, {y})",
            "dig_at_location({x}, {y})", 
            "lift_bucket()",
            "lower_bucket()",
            "turn_left({degrees})",
            "turn_right({degrees})",
            "move_forward({distance})",
            "move_backward({distance})",
            "dump_load()",
            "approach_pile({x}, {y})",
            "position_bucket({x}, {y})",
            "start_digging()",
            "stop_engine()",
            "reverse_to_position({x}, {y})"
        ]
        #  question templates + multi-step commands
        self.question_templates = [
            "Where should the loader go next?",
            "What action should be taken?",
            "Where is the nearest pile?",
            "How should the loader approach the pile?",
            "What is the next step?",
            "Where should the bucket be positioned?",
            "What direction should the loader turn?",
            "Where should the loader dig?",
            "Fill the bucket with material",
            "Go to the nearest pile and load",
            "Move to the dumping area and unload",
            "Position the loader for optimal digging",
            "Complete the loading sequence",
            "Execute the dumping procedure"
        ]
    def generate_caption(self, image_path: str) -> str:
        """Generate a caption for the image"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt")
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return "site with heavy machinery"

    def generate_vqa_pair(self, image_path: str, bbox_data: Dict = None) -> Dict:
        """Generate a VQA pair for a given image with object detection data"""
        
        #base caption
        caption = self.generate_caption(image_path)
        # random question
        question = random.choice(self.question_templates)
        # action based on question and context
        if self._is_multi_step_command(question):
            action = self._generate_action_sequence(question, caption, bbox_data)
        else:
            action = self._generate_action(question, caption, bbox_data)
        return {
            "image_path": image_path,
            "question": question,
            "answer": action,
            "caption": caption,
            "bbox_data": bbox_data or {},
            "action_type": "sequence" if ";" in str(action) else "single"
        }

    def _is_multi_step_command(self, question: str) -> bool:
        """Check if question requires multi-step action sequence"""
        multi_step_keywords = [
            "fill the bucket", "go to the nearest pile and load", 
            "move to the dumping area", "complete the loading",
            "execute the dumping", "position the loader for"
        ]
        return any(keyword in question.lower() for keyword in multi_step_keywords)
    def _generate_action_sequence(self, question: str, caption: str, bbox_data: Dict) -> str:
        """multi-step action sequences for complex commands"""
        if "fill the bucket" in question.lower():
            if bbox_data and bbox_data.get("piles"):
                pile = self._find_nearest_pile(bbox_data["piles"])
                return f"drive_to_pile({pile['x']:.2f}, {pile['y']:.2f}); lower_bucket(); dig_at_location({pile['x']:.2f}, {pile['y']:.2f}); lift_bucket()"
            else:
                x, y = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)
                return f"approach_pile({x:.2f}, {y:.2f}); lower_bucket(); dig_at_location({x:.2f}, {y:.2f}); lift_bucket()"  
        elif "go to the nearest pile and load" in question.lower():
            if bbox_data and bbox_data.get("piles"):
                pile = self._find_nearest_pile(bbox_data["piles"])
                return f"drive_to_pile({pile['x']:.2f}, {pile['y']:.2f}); position_bucket({pile['x']:.2f}, {pile['y']:.2f}); start_digging()"
            else:
                x, y = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
                return f"drive_to_pile({x:.2f}, {y:.2f}); position_bucket({x:.2f}, {y:.2f}); start_digging()"
                
        elif "dumping area" in question.lower():
            # dump area or use random loc
            if bbox_data and bbox_data.get("dump_zones"):
                zone = random.choice(bbox_data["dump_zones"])
                return f"drive_to_position({zone['x']:.2f}, {zone['y']:.2f}); position_for_dump(); dump_load(); move_backward(2.0)"
            else:
                x, y = random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)
                return f"drive_to_position({x:.2f}, {y:.2f}); position_for_dump(); dump_load(); move_backward(2.0)"           
        else:
            # fallback to 1 action
            return self._generate_action(question, caption, bbox_data)

    def _find_nearest_pile(self, piles: List[Dict]) -> Dict:
        """Find the nearest pile"""
        if not piles:
            return {"x": 0.5, "y": 0.5}
        return random.choice(piles)
    def _generate_action(self, question: str, caption: str, bbox_data: Dict) -> str:
        """Generate action based on question and context with obj. det"""
        if "where" in question.lower() and "pile" in question.lower():
            if bbox_data and bbox_data.get("piles"):
                #pile locations
                pile = random.choice(bbox_data["piles"])
                x, y = pile.get("center", (pile.get("x", 0.5), pile.get("y", 0.5)))
                confidence = pile.get("confidence", 0.5)
                # action adjust on confidence
                if confidence > 0.7:
                    return f"drive_to_pile({x:.2f}, {y:.2f})"
                else:
                    return f"approach_pile({x:.2f}, {y:.2f})"  # More cautious approach
            else:
                #fallback random cords
                x, y = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
                return f"drive_to_pile({x:.2f}, {y:.2f})" 
        elif "dig" in question.lower() or "shovel" in question.lower():
            if bbox_data and bbox_data.get("piles"):
                pile = random.choice(bbox_data["piles"])
                x, y = pile.get("center", (pile.get("x", 0.5), pile.get("y", 0.5)))
                return f"dig_at_location({x:.2f}, {y:.2f})"
            else:
                x, y = random.uniform(0.3, 0.7), random.uniform(0.4, 0.8)
                return f"dig_at_location({x:.2f}, {y:.2f})"
        elif "approach" in question.lower() or "go" in question.lower():
            if bbox_data and bbox_data.get("piles"):
                pile = random.choice(bbox_data["piles"])
                x, y = pile.get("center", (pile.get("x", 0.5), pile.get("y", 0.5)))
                return f"approach_pile({x:.2f}, {y:.2f})"
            else:
                x, y = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
                return f"approach_pile({x:.2f}, {y:.2f})"
        elif "turn" in question.lower():
            direction = random.choice(["left", "right"])
            degrees = random.randint(15, 90)
            return f"turn_{direction}({degrees})"
        elif "bucket" in question.lower():
            if "position" in question.lower():
                if bbox_data and bbox_data.get("piles"):
                    pile = random.choice(bbox_data["piles"])
                    x, y = pile.get("center", (pile.get("x", 0.5), pile.get("y", 0.5)))
                    return f"position_bucket({x:.2f}, {y:.2f})"
                else:
                    return random.choice(["lift_bucket()", "lower_bucket()"])
            elif "lift" in question.lower():
                return "lift_bucket()"
            else:
                return "lower_bucket()"
        elif "dump" in question.lower() or "unload" in question.lower():
            return "dump_load()"
        elif "move" in question.lower() or "forward" in question.lower():
            distance = random.uniform(0.5, 3.0)
            return f"move_forward({distance:.1f})"
        elif "backward" in question.lower() or "reverse" in question.lower():
            distance = random.uniform(0.5, 2.0)
            return f"move_backward({distance:.1f})"
        else:
            #fallback using caption and detection data base don the image context
            if bbox_data and bbox_data.get("piles"):
                pile = random.choice(bbox_data["piles"])
                x, y = pile.get("center", (pile.get("x", 0.5), pile.get("y", 0.5)))
                return f"drive_to_pile({x:.2f}, {y:.2f})"
            elif "pile" in caption.lower() or "material" in caption.lower():
                x, y = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
                return f"drive_to_pile({x:.2f}, {y:.2f})"
            elif "dig" in caption.lower() or "excavation" in caption.lower():
                x, y = random.uniform(0.3, 0.7), random.uniform(0.4, 0.8)
                return f"dig_at_location({x:.2f}, {y:.2f})"
            else:
                return "move_forward(1.0)"

def generate_vqa_pairs(data_dir: str):
    """VQA pairs for frames with obj. detection"""
    generator = VQAGenerator()
    try:
        from object_detection import ObjectDetector
        detector = ObjectDetector()
        use_detection = True
        logger.info("Object detection enabled")
    except ImportError:
        logger.warning("Object detection not available, using random coordinates")
        detector = None
        use_detection = False
    #frame paths
    frame_paths_file = Path(data_dir) / "frame_paths.txt"
    if not frame_paths_file.exists():
        logger.error(f"Frame paths file not found: {frame_paths_file}")
        return
    with open(frame_paths_file, 'r') as f:
        frame_paths = [line.strip() for line in f.readlines()]
    vqa_pairs = []
    for i, frame_path in enumerate(frame_paths):
        if i % 10 == 0:
            logger.info(f"Processing frame {i+1}/{len(frame_paths)}")
        if not Path(frame_path).exists():
            logger.warning(f"Frame not found: {frame_path}")
            continue
        # obj. detection data
        bbox_data = {}
        if use_detection and detector:
            try:
                detections = detector.detect_objects(frame_path, confidence_threshold=0.4)
                bbox_data = detections
            except Exception as e:
                logger.warning(f"Detection failed for {frame_path}: {e}")
                bbox_data = {}
            
        #multiple VQA pairs per image
        num_pairs = 3 if bbox_data.get("piles") else 2 
        for _ in range(num_pairs):
            vqa_pair = generator.generate_vqa_pair(frame_path, bbox_data)
            vqa_pairs.append(vqa_pair)
    
    # Save VQA pairs
    output_file = Path(data_dir) / "vqa_pairs.json"
    with open(output_file, 'w') as f:
        json.dump(vqa_pairs, f, indent=2)
    #stats
    single_actions = sum(1 for pair in vqa_pairs if pair.get("action_type") == "single")
    sequences = sum(1 for pair in vqa_pairs if pair.get("action_type") == "sequence")
    
    logger.info(f"Generated {len(vqa_pairs)} VQA pairs:")
    logger.info(f"-Single actions:{single_actions}")
    logger.info(f"-Action sequences:{sequences}")
    logger.info(f"-Saved to:{output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_vqa_pairs("data")