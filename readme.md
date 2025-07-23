
#  Vision-Language-Action (VLA) Model

A functional prototype of a Vision-Language-Action (VLA) system designed to control a robot using natural language commands. It combines computer vision, natural language processing, and pragmatic engineering to produce actionable commands for tasks such as locating and digging material piles on construction sites.

---

## code property

- **Automated Data Pipeline**  
  Downloads POV wheel loader videos from YouTube and automatically extracts frames for dataset creation.

- **Automatic VQA Generation**  
  Uses BLIP captioning, object detection, and predefined templates to generate Visual Question Answering (VQA) pairs.

- **Hybrid Object Detection**  
  Combines:
  - **OWL-ViT** for open-vocabulary detection  
  - **YOLOv8 + CLIP** for fast and specific classification  
  - **OpenCV** for color-based detection of dirt piles

- **Synthetic Pile Detection**  
  If no piles are detected, synthetic ones are generated to ensure consistent training examples.

- **Efficient LLM Fine-Tuning**  
  Fine-tunes `unsloth/Llama-3.2-1B-Instruct` using **LoRA** for parameter-efficient learning.

- **Robust Inference Pipeline**  
  Includes multi-step command parsing and rule-based fallback logic to ensure reliable action prediction.

- **Interactive Demo**  
  A Gradio UI for uploading images, entering commands, and viewing predicted actions and visual annotations.

---

## Workflow

```
[1. Data Prep] --> [2. VQA Generation] --> [3. Model Training]
YouTube → Download → Extract Frames → Object Detection + Captioning → VQA Pairs → PEFT LLM (parameter efficient fine tuning LLM)

[4. Inference & Demo]
Command + Image → Vision Module → Detections → Prompt Formatter → LLM → Parsed Action
```

---


### 1. Setup

```bash

pip install -r requirements.txt
```

---

### 2. Pipeline: Data Prep & Training

**Step 1: Prepare the Dataset**

Downloads videos, extracts frames, detects objects, and generates VQA pairs.

```bash
python main.py --mode prepare --max_videos 5 --frames_per_video 50
```

**Step 2: Train the Model**

Fine-tunes the LLaMA-3.2-1B model from unsloth on generated data.

```bash
python main.py --mode train
```

---

### 3. Run the Interactive Demo

Once training is complete and the `final_model/` directory is available:

```bash
python main.py --mode gradio
```
Or just go to the google colab link and run code here:https://colab.research.google.com/drive/1wQmVS64llr6V4h_M7XgOcem7QWf-6k6i?usp=sharing
---

## Technical Details

### Hybrid Object Detection Strategy

| Detector     | Strengths                                   | Use Case                              |
|--------------|---------------------------------------------|----------------------------------------|
| YOLOv8+CLIP  | Fast, general proposals + text classification | Base detection + class disambiguation |
| OWL-ViT      | Open vocabulary queries, robust to unknowns | Secondary fallback for missed objects |
| OpenCV       | Simple, fast pile detection via color masks | Tertiary fallback                     |

> When no piles are detected, synthetic locations are generated to ensure consistent data for the LLM.

---

### Prompt Engineering for foundation models

Structured input to LLM:

```text
### Visual Context:
Visual analysis shows 2 material pile(s). Pile 1: located at (0.75, 0.60)...
### Operator Command:
Go to the nearest pile.
### Required Action:
Action:
```

---

### Action Space & Fallback Logic

- **Primary Actions**: `drive_to_pile(x,y)`, `lift_bucket()`, `dump_load()`, etc.
- **Fallback Logic**: If LLM output is unparseable, rule-based parsing checks for keywords like `dump`, `fill`, or `move` to trigger default safe actions.

---

## Limitations

- **Speed**: Multiple large models lead to high latency, but needs optimization.
- **Detection Sensitivity**: Detection thresholds must balance false positives vs. missed objects.
- **LLM Reasoning**: May occasionally infer incorrect actions due to ambiguous prompts or lack of visual grounding.
- **Data Bias**: Limited YouTube videos (I just trained on 3 videos), so limited generalization.

---

## possiblities

- **Better Prompting**: Add chain-of-thought or structured reasoning steps.
- **Temporal Context**: Use short video clips instead of single frames.
- **Expanded Action Set**: More compound actions.
- **End-to-End Models**: Explore fully integrated VLA models like UniVLA.
- **Performance Optimization**: Quantization, batching, and inference acceleration.

---

## Acknowledgements for code reuse and repurpose

This work is inspired by and builds (read code repurpose and refinement) on below projects:

- [`zhaw-iwi/MultimodalInteraction_ObjDet`](https://github.com/zhaw-iwi/MultimodalInteraction_ObjDet)
- [`stevebottos/owl-vit-object-detection`](https://github.com/stevebottos/owl-vit-object-detection)
- [`ShwetaTyagi1/VideoProcessing_and_FrameExtraction`](https://github.com/ShwetaTyagi1/VideoProcessing_and_FrameExtraction)
- [`baaivision/UniVLA`](https://github.com/baaivision/UniVLA)
- [`huynhtinn/VQA_project`](https://github.com/huynhtinn/VQA_project)
- [`kumarmanas/SHIFT`](https://github.com/kumarmanas/SHIFT)


