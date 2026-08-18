import os
import sys

import gradio as gr
import torch

from src.inference import load_pipeline, predict, draw_boxes

CHECKPOINT_DIR = os.environ.get("FRCNN_CHECKPOINT_DIR", "checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    pipeline = load_pipeline(CHECKPOINT_DIR, DEVICE)
except FileNotFoundError as e:
    print(f"Failed to load model: {e}", file=sys.stderr)
    sys.exit(1)


def detect(pil_image, score_thresh, nms_iou_thresh):
    detections = predict(pipeline, pil_image, score_thresh, nms_iou_thresh)
    return draw_boxes(pil_image, detections)


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Score threshold"),
        gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="NMS IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Detections"),
    title="Faster R-CNN (Pascal VOC)",
)

if __name__ == "__main__":
    demo.launch()
