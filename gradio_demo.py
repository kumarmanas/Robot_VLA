#!/usr/bin/env python3
"""
Gradio VLA model demo
"""
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
import logging

from inference import WheelLoaderInference
from object_detection import ObjectDetector

logger = logging.getLogger(__name__)

def draw_bounding_boxes(image, detections):
    """Draw bounding boxes on image"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    #image dimensions
    img_width = image.width
    img_height = image.height
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    # piles (green rectangles with bbox)
    for i, pile in enumerate(detections.get("piles", [])):
        if "bbox" in pile and pile["bbox"] != [0, 0, 100, 100]:  #ignore invalid bbox
            bbox = pile["bbox"]
            # check if bbox is normalized or not (values<=1.0) or in pix
            if all(coord <= 1.0 for coord in bbox) and all(coord >= 0.0 for coord in bbox):
                #normalized bbox- convert to pixs
                x1 = int(bbox[0] * img_width)
                y1 = int(bbox[1] * img_height)
                x2 = int(bbox[2] * img_width)
                y2 = int(bbox[3] * img_height)
                #check coords within image bounds
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                # draw and label for valid rectangle
                if x2 > x1 and y2 > y1:
                    draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                    confidence = pile.get("confidence", 0.0)
                    label = f"Pile {i+1}: {confidence:.2f}"
                    draw.text((x1, max(0, y1-25)), label, fill="lime", font=font)
                    continue
        # Fallback to center point for invalid/missing bbox
        center = pile.get("center", [0.5, 0.5])
        x = int(center[0] * img_width)
        y = int(center[1] * img_height)
        w, h = 60, 60
        draw.rectangle([x-w//2, y-h//2, x+w//2, y+h//2], 
                      outline="lime", width=3)
        confidence = pile.get("confidence", 0.0)
        is_synthetic = "synthetic" in pile.get("query", "")
        label = f"{'Synth ' if is_synthetic else ''}Pile {i+1}: {confidence:.2f}"
        draw.text((x-30, max(0, y-40)), label, fill="lime", font=font)
    
    # Draw vehicles(cyan color)
    for i, vehicle in enumerate(detections.get("vehicles", [])):
        if "bbox" in vehicle:
            bbox = vehicle["bbox"]
            #check if bbox is normalized (values<=1.0) or in pixs and then normalize and preprocessing
            if all(coord <= 1.0 for coord in bbox) and all(coord >= -0.1 for coord in bbox):  # Allow slight negative values
                x1 = int(bbox[0] * img_width)
                y1 = int(bbox[1] * img_height)
                x2 = int(bbox[2] * img_width)
                y2 = int(bbox[3] * img_height)
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                if x2 > x1 and y2 > y1:
                    draw.rectangle([x1, y1, x2, y2], outline="cyan", width=3)
                    #box label
                    confidence = vehicle.get("confidence", 0.0)
                    query = vehicle.get("query", "vehicle")
                    label = f"{query}: {confidence:.2f}"
                    draw.text((x1, max(0, y1-25)), label, fill="cyan", font=font)
                    continue
        # Fallback if invalid boundind box then rectangle around center
        center = vehicle.get("center", [0.5, 0.5])
        x = int(center[0] * img_width)
        y = int(center[1] * img_height)
        w, h = 80, 60
        draw.rectangle([x-w//2, y-h//2, x+w//2, y+h//2], 
                      outline="cyan", width=3)
        confidence = vehicle.get("confidence", 0.0)
        query = vehicle.get("query", "vehicle")
        label = f"{query}: {confidence:.2f}"
        draw.text((x-40, max(0, y-35)), label, fill="cyan", font=font)
    return np.array(img_with_boxes)

def format_multi_step_action(action: str) -> str:
    """multi-step actions"""
    if ";" in action:
        steps = action.split(";")
        formatted_steps = []
        for i, step in enumerate(steps):
            formatted_steps.append(f"   {i+1}. {step.strip()}")
        return "**Multi-Step Sequence:**\n" + "\n".join(formatted_steps)
    else:
        return f"`{action}`"
    
def filter_detections_by_confidence(detections, confidence_threshold=0.4):
    """Filter detections based on confidence threshold"""
    filtered_detections = {}
    
    for category in ["piles", "vehicles", "excavation_areas"]:
        original_items = detections.get(category, [])
        filtered_items = [
            item for item in original_items 
            if item.get("confidence", 0.0) >= confidence_threshold
        ]
        filtered_detections[category] = filtered_items
        
        if len(original_items) != len(filtered_items):
            logger.info(f"Filtered {category}: {len(original_items)} -> {len(filtered_items)} "
                       f"(removed {len(original_items) - len(filtered_items)} low-confidence items)")
    return filtered_detections

def evaluate_wheel_loader_command(image, command, confidence_threshold):
    """
    Main evaluation function
    """
    if image is None:
        return "Upload an image", None
    
    if not command or not command.strip():
        return "Enter a command", image
    
    try:
        detector = ObjectDetector()
        inference_system = WheelLoaderInference("./models")
        temp_path = "temp_demo_image.jpg"
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(temp_path)
        else:
            image.save(temp_path)
        #obj. det. with the confidence threshold and filter them out
        detections = detector.detect_objects(temp_path, confidence_threshold=confidence_threshold)  # Changed this line
        filtered_detections = filter_detections_by_confidence(detections, confidence_threshold)
        #inference for action
        action, _ = inference_system.process_command(temp_path, command, filtered_detections)
        #image annotation with filtered det.
        annotated_image = draw_bounding_boxes(image, filtered_detections)
        pile_count = len(filtered_detections.get("piles", []))
        vehicle_count = len(filtered_detections.get("vehicles", []))
        excavation_count = len(filtered_detections.get("excavation_areas", []))
        action_display = format_multi_step_action(action)
        response = f"""## VLA Response

**Command:** {command}

**Recommended Action:** 
{action_display}

**Detection Summary (≥{confidence_threshold*100:.0f}% confidence):**

- Material Piles: {pile_count}
- Vehicles: {vehicle_count}  
- Excavation Areas: {excavation_count}

**Only high-Confidence Pile Locations:**"""
        
        if pile_count > 0:
            for i, pile in enumerate(filtered_detections["piles"][:5]): #only 5 files max
                center = pile.get("center", [0, 0])
                confidence = pile.get("confidence", 0)
                response += f"\n- Pile {i+1}: ({center[0]:.2f}, {center[1]:.2f}) [confidence: {confidence:.1%}]"
        else:
            response += "\n- No high-confidence piles detected"
        Path(temp_path).unlink(missing_ok=True)
        return response, annotated_image
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return f"Error: {str(e)}", image

def create_simple_demo():
    # Example commands
    example_commands = [
        "Move to the dumping area and unload", 
        "Move to the material pile",
        "Where should I dig?",
        "Move to the dumping area and unload", 
        "Go to the next pile",
        "Position the bucket for loading",
        "What action should be taken?"
    ]
    
    with gr.Blocks(title="Wheel Loader VLA", theme=gr.themes.Default()) as demo:
        
        gr.Markdown("""
        # demo wheel loader VLA
        
        Upload a construction site image and give commands and it will show object detection (select obj. det model threshold below) and action.
        """)
        
        with gr.Row():
            #I/p column
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Construction Site Bild",
                    type="numpy"
                )
                
                command_input = gr.Textbox(
                    label="user command input",
                    placeholder="ex: Move to the material pile",
                    lines=2
                )
                #confidence threshold selection slider
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.4,
                    step=0.05,
                    label="Confidence Threshold",
                    info="user defined obj. det. confidence interval selection"
                )
                submit_btn = gr.Button("Execute Command", variant="primary", size="lg")
                
                gr.Markdown("###example command:")
                example_buttons = []
                for cmd in example_commands:
                    btn = gr.Button(f'"{cmd}"', size="sm")
                    btn.click(lambda x=cmd: x, outputs=command_input)
                    example_buttons.append(btn)
            
            # Output column  
            with gr.Column(scale=1):
                response_output = gr.Markdown(
                    value="Upload an image and command",
                    label="AI Response"
                )
                
                annotated_image = gr.Image(
                    label="Obj. Det. Results",
                    type="numpy"
                )
        submit_btn.click(
            fn=evaluate_wheel_loader_command,
            inputs=[image_input, command_input,confidence_slider],
            outputs=[response_output, annotated_image]
        )
        
        gr.Markdown("""
        ---
        ### Legend:
        - 🟢 **Green**: Material piles 🔵 **Cyan**: Construction vehicles 🟠 **Orange**: Excavation areas
        **Coordinates normalized in range 0.0 to 1.0**
        """)
    return demo

def main():
    """Launch demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Wheel Loader Demo")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    if not Path("./models/final_model").exists():
        print("No trained model found")
        print("Run: python main.py --mode train")
        return
    
    print("Launch demo")
    demo = create_simple_demo()
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0" if args.share else "127.0.0.1"
    )
if __name__ == "__main__":
    main()