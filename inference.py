"""
Inference pipeline for the wheel loader action model
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image
import logging
from pathlib import Path
from object_detection import ObjectDetector
import random
import re

logger = logging.getLogger(__name__)

class MultiStepCommandHandler:
    def __init__(self):
        self.command_patterns = {
            "pile_and_load": {
                "keywords": ["go to the nearest pile and load", "pile and load", "load from pile"],
                "actions": ["drive_to_pile", "position_bucket", "start_digging"],
                "requires_pile": True
            },
            "fill_bucket": {
                "keywords": ["fill the bucket", "fill bucket", "load bucket"],
                "actions": ["drive_to_pile", "lower_bucket", "dig_at_location", "lift_bucket"],
                "requires_pile": True
            },
            "dump_sequence": {
                "keywords": [
                    "move to the dumping area and unload",
                    "move to dumping area and unload", 
                    "dumping area and unload",
                    "go to dumping area and unload",
                    "drive to dumping area and unload",
                    "unload at dumping area",
                    "dump load at dumping area"
                ],
                "actions": ["drive_to_position", "position_for_dump", "dump_load", "move_backward"],
                "requires_pile": False,
                "default_params": {"distance": 2.0}
            }
        }
    
    def match_command(self, command: str) -> str:
        """Match command to pattern"""
        command_lower = command.lower()
        for pattern_name, pattern in self.command_patterns.items():
            if any(keyword in command_lower for keyword in pattern["keywords"]):
                return pattern_name
        return None
    
    def generate_sequence(self, pattern_name: str, detections: dict, pile_selector_func) -> str:
        """Generate action sequence for pattern"""
        pattern = self.command_patterns[pattern_name]
        actions = []
        
        # Get location parameters
        if pattern.get("requires_pile") and detections.get("piles"):
            pile = pile_selector_func(detections["piles"], "digging")
            x, y = pile.get("center", [0.5, 0.5])
        else:
            x, y = self._get_default_position(pattern_name)
        
        # Build action sequence
        for action in pattern["actions"]:
            formatted_action = self._format_action(action, x, y, pattern.get("default_params", {}))
            actions.append(formatted_action)
        
        return "; ".join(actions)
    
    def _get_default_position(self, pattern_name: str):
        """Get default position for pattern type"""
        if pattern_name == "dump_sequence":
            return 0.2, 0.2  # Dumping area
        else:
            return 0.5, 0.6  # General pile area
    
    def _format_action(self, action_name: str, x: float, y: float, params: dict) -> str:
        """Format action with parameters"""
        action_templates = {
            "drive_to_pile": f"drive_to_pile({x:.2f}, {y:.2f})",
            "drive_to_position": f"drive_to_position({x:.2f}, {y:.2f})",
            "approach_pile": f"approach_pile({x:.2f}, {y:.2f})",
            "position_bucket": f"position_bucket({x:.2f}, {y:.2f})",
            "position_for_dump": f"position_for_dump({x:.2f}, {y:.2f})", 
            "dig_at_location": f"dig_at_location({x:.2f}, {y:.2f})",
            "start_digging": f"start_digging({x:.2f}, {y:.2f})",
            "lift_bucket": "lift_bucket()", 
            "lower_bucket": "lower_bucket()", 
            "dump_load": f"dump_load({x:.2f}, {y:.2f})",
            "move_backward": f"move_backward({params.get('distance', 1.5):.1f})",
            "move_forward": f"move_forward({params.get('distance', 1.0):.1f})"
        }
        
        return action_templates.get(action_name, f"{action_name}()")
     
class WheelLoaderInference:
    def __init__(self, model_dir: str):
        """Initialize the inference pipeline"""
        self.model_dir = Path(model_dir)
        self.multi_step_handler = MultiStepCommandHandler()
        final_model_path = self.model_dir / "final_model"
        config_path = final_model_path / "adapter_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                base_model_name = config.get("base_model_name_or_path", "unsloth/Llama-3.2-1B-Instruct")
            
            # Load base model
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
            # Load fine-tuned model
            self.model = PeftModel.from_pretrained(base_model, final_model_path)
            logger.info("Loaded fine-tuned model")
        else:
            base_model_name = "unsloth/Llama-3.2-1B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
            logger.warning("No FT model so loaded base model")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.detector = ObjectDetector()
        self.model.eval()

    def process_command(self, image_path: str, command: str, detections: dict = None):
        """command process via image context"""
        
        try:
            #Get obj. det
            if detections is None:
                detections = self.detector.detect_objects(image_path)
            
            #check obj det format
            if not isinstance(detections, dict):
                detections = {"piles": [], "vehicles": [], "excavation_areas": []}
                
            #prompt with image context
            prompt = self._create_prompt(command, detections)
            
            #response with proper token
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=inputs["input_ids"].shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Extract action from response
            action = self._extract_action(response, command, detections)
            
            return action, detections
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            # Fallback to basic action
            return self._generate_fallback_action(command, detections or {}), detections or {}

    def _create_prompt(self, command: str, detections: dict) -> str:
        """Create a prompt with visual context"""
        
        # Build context from detections
        context = []
        if detections.get("piles"):
            pile_count = len(detections["piles"])
            context.append(f"Visual analysis shows {pile_count} material pile(s)")
            
            # Add specific pile locations
            for i, pile in enumerate(detections["piles"][:3]):
                x, y = pile.get("center", (pile.get("x", 0.5), pile.get("y", 0.5)))
                conf = pile.get("confidence", 0.5)
                context.append(f"Pile {i+1}: located at ({x:.2f}, {y:.2f}) with {conf:.1%} confidence")
        
        if detections.get("vehicles"):
            vehicle_count = len(detections["vehicles"])
            context.append(f"Detected {vehicle_count} construction vehicle(s)")
        
        # Enhanced prompt structure
        context_str = ". ".join(context) if context else "Construction site analysis in progress"
        
        prompt = f"""## Wheel Loader Operation Assistant

### Visual Context:
{context_str}

### Operator Command: 
{command}

### Required Action:
Generate the appropriate wheel loader control command. Use format like:
- drive_to_pile(x, y) for navigation
- dig_at_location(x, y) for digging
- lift_bucket() for bucket control
- approach_pile(x, y) for careful approach

Action:"""

        return prompt

    def _extract_action(self, response: str, command: str, detections: dict) -> str:
        """Enhanced extraction with natural language support"""
        
        response_clean = response.strip().lower()
        command_lower = command.lower()
        pattern_name = self.multi_step_handler.match_command(command)
        if pattern_name:
            result = self.multi_step_handler.generate_sequence(pattern_name, detections, self._select_best_pile_for_command)
            return result
        action_patterns = [
            r'(drive_to_pile|approach_pile|dig_at_location)\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)',
            r'(turn_left|turn_right)\s*\(\s*(\d+)\s*\)',
            r'(lift_bucket|lower_bucket|dump_load)\s*\(\s*\)'
        ]
        for pattern in action_patterns:
            match = re.search(pattern, response_clean)
            if match:
                action_type = match.group(1)
                if action_type in ['drive_to_pile', 'approach_pile', 'dig_at_location']:
                    try:
                        x = float(match.group(2))
                        y = float(match.group(3))
                        x = max(0.0, min(1.0, x))
                        y = max(0.0, min(1.0, y))
                        return f"{action_type}({x:.2f}, {y:.2f})"
                    except (ValueError, IndexError):
                        continue
                else:
                    return match.group(0)
        
        # Parse response
        coordinates = re.findall(r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', response_clean)
        if coordinates:
            x, y = float(coordinates[0][0]), float(coordinates[0][1])
            x, y = max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
            if any(word in command.lower() for word in ["move", "go", "drive"]):
                return f"drive_to_pile({x:.2f}, {y:.2f})"
            elif any(word in command.lower() for word in ["fill", "dig"]):
                return f"approach_pile({x:.2f}, {y:.2f})"
        if "lift" in response_clean and "bucket" in response_clean:
            return "lift_bucket()"
        elif "lower" in response_clean and "bucket" in response_clean:
            return "lower_bucket()"
        elif "dump" in response_clean and not any(word in command_lower for word in ["dumping area", "unload"]):
            return "dump_load()"
        return self._generate_command_specific_action(command, detections)

    def _generate_command_specific_action(self, command: str, detections: dict) -> str:
        """Generate command-specific actions based on training data patterns"""
        pattern_name = self.multi_step_handler.match_command(command)
        if pattern_name:
            result = self.multi_step_handler.generate_sequence(pattern_name, detections, self._select_best_pile_for_command)
            logger.info(f"Generated sequence: {result}")
            return result
        command_lower = command.lower()
        # movement/nav command
        if any(word in command_lower for word in ["move to", "go to", "drive to", "navigate"]):
            if detections.get("piles"):
                pile = self._select_best_pile_for_command(detections["piles"], "navigation")
                x, y = pile.get("center", [0.5, 0.5])
                return f"drive_to_pile({x:.2f}, {y:.2f})"
            else:
                x, y = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)
                return f"drive_to_pile({x:.2f}, {y:.2f})"
        
        # fil/dig command  
        elif any(word in command_lower for word in ["fill", "shovel", "scoop", "load material"]):
            if detections.get("piles"):
                pile = self._select_best_pile_for_command(detections["piles"], "digging")
                x, y = pile.get("center", [0.5, 0.5])
                if "fill" in command_lower:
                    return f"approach_pile({x:.2f}, {y:.2f})"
                else:
                    return f"dig_at_location({x:.2f}, {y:.2f})"
            else:
                x, y = random.uniform(0.3, 0.7), random.uniform(0.4, 0.8)
                return f"dig_at_location({x:.2f}, {y:.2f})"
        
        # bucket control 
        elif any(word in command_lower for word in ["bucket", "lift", "raise", "lower"]):
            if "lift" in command_lower or "raise" in command_lower or "up" in command_lower:
                return "lift_bucket()"
            else:
                return "lower_bucket()"
        
        # dumping command
        elif any(word in command_lower for word in ["dump", "unload", "empty"]):
            return "dump_load()"
        #turn command
        elif any(word in command_lower for word in ["turn", "rotate"]):
            direction = "left" if "left" in command_lower else random.choice(["left", "right"])
            degrees = random.randint(20, 60)
            return f"turn_{direction}({degrees})"
        # general action ques
        elif any(word in command_lower for word in ["action", "next", "should", "what"]):
            return self._generate_contextual_action(detections)
        
        # default fallback
        else:
            if detections.get("piles"):
                pile = detections["piles"][0]
                x, y = pile.get("center", [0.5, 0.5])
                return f"approach_pile({x:.2f}, {y:.2f})"
            else:
                return "move_forward(1.0)"

    def _select_best_pile_for_command(self, piles: list, command_type: str) -> dict:
        """Select the most appropriate pile based on command type"""
        
        if not piles:
            return {"center": [0.5, 0.5]}
        
        if command_type == "navigation":
            # For navigation, prefer higher confidence piles
            return max(piles, key=lambda p: p.get("confidence", 0))
        
        elif command_type == "digging":
            # For digging, prefer piles in accessible locations
            accessible_piles = [p for p in piles if p.get("center", [0, 0.5])[1] > 0.4]
            if accessible_piles:
                return max(accessible_piles, key=lambda p: p.get("confidence", 0))
            else:
                return piles[0]
        
        return piles[0]

    def _generate_contextual_action(self, detections: dict) -> str:
        """Generate contextual actions for general questions based on scene"""
        
        action_types = ["bucket_operations", "movement", "pile_operations", "turning"]
        action_type = random.choice(action_types)
        
        if action_type == "bucket_operations":
            return random.choice(["lift_bucket()", "lower_bucket()", "dump_load()"])
        
        elif action_type == "movement":
            distance = random.uniform(0.5, 3.0)
            direction = random.choice(["forward", "backward"])
            return f"move_{direction}({distance:.1f})"
        
        elif action_type == "turning":
            direction = random.choice(["left", "right"])
            degrees = random.randint(15, 90)
            return f"turn_{direction}({degrees})"
        
        else:  # pile_operations
            if detections.get("piles"):
                pile = random.choice(detections["piles"])
                x, y = pile.get("center", [0.5, 0.5])
                action = random.choice(["drive_to_pile", "approach_pile", "dig_at_location"])
                return f"{action}({x:.2f}, {y:.2f})"
            else:
                x, y = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)
                return f"approach_pile({x:.2f}, {y:.2f})"

    def _generate_fallback_action(self, command: str, detections: dict) -> str:
        """Generate fallback action when model response is unclear"""
        
        command_lower = command.lower()
        
        # Use detected pile locations when available
        if detections.get("piles") and any(keyword in command_lower for keyword in ["pile", "dig", "load", "fill"]):
            pile = detections["piles"][0]
            x, y = pile.get("center", [0.5, 0.5])
            
            if "fill" in command_lower or "load" in command_lower:
                return f"approach_pile({x:.2f}, {y:.2f})"
            elif "dig" in command_lower:
                return f"dig_at_location({x:.2f}, {y:.2f})"
            else:
                return f"drive_to_pile({x:.2f}, {y:.2f})"
        
        # Command-based fallbacks
        elif "bucket" in command_lower:
            if "lift" in command_lower or "up" in command_lower:
                return "lift_bucket()"
            else:
                return "lower_bucket()"
        elif "dump" in command_lower:
            return "dump_load()"
        elif "turn" in command_lower:
            direction = "left" if "left" in command_lower else "right"
            degrees = random.randint(15, 45)
            return f"turn_{direction}({degrees})"
        else:
            # Random pile approach as last resort
            x, y = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)
            return f"approach_pile({x:.2f}, {y:.2f})"
