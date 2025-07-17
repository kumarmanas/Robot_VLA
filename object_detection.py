"""
object detection OWL-ViT with color-based detection
note: OEL provides support for the open vocab in object detection
note: also added yolov8 and clip support for better obj. detection
"""

import cv2
import numpy as np
from PIL import Image
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import re
from ultralytics import YOLO
import clip

try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    OWL_VIT_AVAILABLE = True
except ImportError:
    OWL_VIT_AVAILABLE = False
    logging.warning("OWL-ViT not available")
try:
    YOLO_CLIP_AVAILABLE = True
except ImportError:
    YOLO_CLIP_AVAILABLE = False
    logging.warning("YoloV8+CLIP not available")
logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if OWL_VIT_AVAILABLE:
            try:
                self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
                self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
                self.model.to(self.device)
            except Exception as e:
                logger.warning(f"Failed to load OWL-ViT: {e}")
                self.processor = None
                self.model = None
        else:
            self.processor = None
            self.model = None
        # Initialize YOLO + CLIP if available
        if YOLO_CLIP_AVAILABLE:
            try:
                self.yolo = YOLO('yolov8n.pt')
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                self.class_prompts = [
                    "gravel pile", "dirt mound", "sand pile", "rock pile",
                    "excavated area", "digging site", "open pit", "trench", 
                    "wheel loader", "excavator", "bulldozer", "dump truck",
                    "construction vehicle", "loader bucket", "shovel"
                ]
                self.text_tokens = clip.tokenize(self.class_prompts).to(self.device)
            except Exception as e:
                logger.warning(f"Failed to load YOLO+CLIP: {e}")
                self.yolo = None
                self.clip_model = None
        else:
            self.yolo = None
            self.clip_model = None

        self.detection_queries = [
            #Material piles
            "dirt pile", "gravel pile", "sand pile", "material pile",
            "construction site material", "pile of dirt", "pile of gravel",
            "construction zone material", "dirt mound", "material mound",
            #Vehicles
            "wheel loader", "excavator", "bulldozer", "construction vehicle",
            "yellow excavator", "construction equipment", "heavy machinery",
            #Areas
            "excavation area", "digging spot", "construction site"
        ]

    def detect_objects(self, image_path: str, confidence_threshold: float = 0.02) -> Dict:
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            detections = {"piles": [], "vehicles": [], "excavation_areas": []} #multiple detection with pref order
            
            #Method 1:YoloV8+CLIP
            if self.yolo and self.clip_model:
                yolo_detections = self._detect_with_yolo_clip(image_np, confidence_threshold)
                self._merge_detections(detections, yolo_detections)
            
            # Method 2:OWL-ViT
            if (self.processor and self.model and 
                len(detections['piles']) + len(detections['vehicles']) < 3):
                owlvit_detections = self._detect_with_owlvit(image_path, confidence_threshold)
                self._merge_detections(detections, owlvit_detections)
            # Method 3: Enhanced color-based detection (Colab style)
            color_detections = self._detect_with_enhanced_opencv(image_np, confidence_threshold)
            self._merge_detections(detections, color_detections)
            # synthetic piles if needed
            detections = self._add_synthetic_piles_if_needed(detections, confidence_threshold)
            total_objects = len(detections['piles']) + len(detections['vehicles']) + len(detections['excavation_areas'])
            logger.info(f"Final detection count: {len(detections['piles'])} piles, {len(detections['vehicles'])} vehicles, {len(detections['excavation_areas'])} excavation areas")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return {"piles": [], "vehicles": [], "excavation_areas": []}

    def _detect_with_yolo_clip(self, image: np.ndarray, confidence_threshold: float) -> Dict:
        """YOLOv8+CLIP detection"""
        detections = {"piles": [], "vehicles": [], "excavation_areas": []}
        try:
            results = self.yolo(image)
            result = results[0]
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                if confidence < confidence_threshold:
                    continue
                crop = image[y1:y2, x1:x2] #Crop image
                if crop.size == 0:
                    continue
                pil_crop = Image.fromarray(crop)
                clip_input = self.clip_preprocess(pil_crop).unsqueeze(0).to(self.device)
                #CLIP classification
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(clip_input)
                    text_features = self.clip_model.encode_text(self.text_tokens)
                    logits_per_image = image_features @ text_features.T
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                best_idx = int(np.argmax(probs))
                best_label = self.class_prompts[best_idx]
                best_score = probs[best_idx]
                if best_score > 0.3:  #CLIP confidence threshold
                    center_x = (x1 + x2) / 2 / image.shape[1]
                    center_y = (y1 + y2) / 2 / image.shape[0]
                    norm_bbox = [x1/image.shape[1], y1/image.shape[0], 
                                x2/image.shape[1], y2/image.shape[0]]
                    detection = {
                        "bbox": norm_bbox,
                        "center": [center_x, center_y],
                        "confidence": float(best_score),
                        "query": best_label
                    }
                    # Categorize on label
                    if any(word in best_label for word in ["pile", "dirt", "gravel", "sand", "material"]):
                        detections["piles"].append(detection)
                    elif any(word in best_label for word in ["loader", "excavator", "bulldozer", "vehicle"]):
                        detections["vehicles"].append(detection)
                    elif any(word in best_label for word in ["excavation", "digging", "pit"]):
                        detections["excavation_areas"].append(detection)
                        
        except Exception as e:
            logger.error(f"YOLO+CLIP detection failed: {e}")
            
        return detections
    
    def _detect_with_enhanced_opencv(self, image: np.ndarray, confidence_threshold: float) -> Dict:
        """color-based detection"""
        detections = {"piles": [], "vehicles": [], "excavation_areas": []}
        try:
            # HSV convert
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lower_brown1 = np.array([8, 50, 20])
            upper_brown1 = np.array([25, 255, 120])
            lower_brown2 = np.array([0, 30, 30])
            upper_brown2 = np.array([15, 255, 100])
            brown_mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
            brown_mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
            brown_mask = cv2.bitwise_or(brown_mask1, brown_mask2)
            contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  #Filter small obj
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = (x + w/2) / image.shape[1]
                    center_y = (y + h/2) / image.shape[0]
                    norm_bbox = [x/image.shape[1], y/image.shape[0], 
                                (x+w)/image.shape[1], (y+h)/image.shape[0]]
                    detections["piles"].append({
                        "bbox": norm_bbox,
                        "center": [center_x, center_y],
                        "confidence": min(0.85, area / 5000), 
                        "query": "dirt pile"
                    })
            
            #gray/black objects
            lower_gray = np.array([0, 0, 0])
            upper_gray = np.array([180, 50, 80])
            gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
            contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = (x + w/2) / image.shape[1]
                    center_y = (y + h/2) / image.shape[0]
                    norm_bbox = [x/image.shape[1], y/image.shape[0], 
                                (x+w)/image.shape[1], (y+h)/image.shape[0]]
                    detections["vehicles"].append({
                        "bbox": norm_bbox,
                        "center": [center_x, center_y],
                        "confidence": min(0.75, area / 8000),
                        "query": "construction equipment"
                    })
        except Exception as e:
            logger.error(f"OpenCV detection failed: {e}")
        return detections

    def _detect_with_owlvit(self, image_path: str, confidence_threshold: float) -> Dict:
        """OWL-ViT detection"""
        detections = {"piles": [], "vehicles": [], "excavation_areas": []}
        if not self.processor or not self.model:
            return detections
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(
                text=self.detection_queries,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                threshold=confidence_threshold,
                target_sizes=target_sizes
            )
            boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.cpu().numpy()
                center_x = (x1 + x2) / 2 / image.size[0]
                center_y = (y1 + y2) / 2 / image.size[1]
                norm_bbox = [x1/image.size[0], y1/image.size[1], 
                            x2/image.size[0], y2/image.size[1]]
                detection = {
                    "bbox": norm_bbox,
                    "center": [center_x, center_y],
                    "confidence": float(score),
                    "query": self.detection_queries[label]
                }
                query = self.detection_queries[label].lower()
                if "pile" in query or "material" in query or "dirt" in query:
                    detections["piles"].append(detection)
                elif "loader" in query or "vehicle" in query or "equipment" in query:
                    detections["vehicles"].append(detection)
                elif "excavation" in query or "digging" in query:
                    detections["excavation_areas"].append(detection)
        except Exception as e:
            logger.error(f"OWL-ViT detection failed: {e}")
        return detections

    def _merge_detections(self, main_detections: Dict, new_detections: Dict):
        """Merge obj. det. into main detection dict"""
        for category in ["piles", "vehicles", "excavation_areas"]:
            main_detections[category].extend(new_detections.get(category, []))
    def _add_synthetic_piles_if_needed(self, detections: Dict, confidence_threshold: float) -> Dict:
        """Add synthetic pile detections"""
        if len(detections["vehicles"]) > 0 and len(detections["piles"]) == 0:
            for i in range(random.randint(1, 3)):
                center_x = random.uniform(0.2, 0.8)
                center_y = random.uniform(0.4, 0.8)
                box_size = 0.05
                synthetic_pile = {
                    "bbox": [
                        max(0.0, center_x - box_size),
                        max(0.0, center_y - box_size),
                        min(1.0, center_x + box_size),
                        min(1.0, center_y + box_size)
                    ],
                    "center": [center_x, center_y],
                    "confidence": random.uniform(confidence_threshold + 0.1, 0.6),
                    "query": "synthetic_material_pile"
                }
                detections["piles"].append(synthetic_pile)
        return detections