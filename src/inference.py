from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import ImageDraw, ImageFont

from src.backbone import Backbone, backbone_transform
from src.rpn import RPN_Head, RegionProposalNetwork
from src.roi import RoIPool
from src.detection_net import DetectionHead, DetectionNet
from src.dataset import VOC_CLASSES

from huggingface_hub import hf_hub_download

# One fixed, distinguishable color per VOC class, keyed by VOC_CLASSES order.
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
]
_CLASS_COLORS = dict(zip(VOC_CLASSES, _PALETTE))


@dataclass
class Pipeline:
    backbone: Backbone
    rpn_network: RegionProposalNetwork
    roi_pool: RoIPool
    detection_network: DetectionNet
    device: torch.device


_CHECKPOINT_NAME = "faster_rcnn_final.bin"


def load_pipeline(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / _CHECKPOINT_NAME).exists():
        print(f"Missing checkpoint file in '{checkpoint_dir}': {_CHECKPOINT_NAME}. ")

        checkpoint_path = hf_hub_download(repo_id="0Curious0/faster_rcnn_resnet50", filename="checkpoints/faster_rcnn_final.bin")
    else:
        checkpoint_path = checkpoint_dir / _CHECKPOINT_NAME

    backbone = Backbone().to(device)
    rpn_head = RPN_Head(in_channels=1024, mid_channels=512)
    detection_head = DetectionHead()

    unified_ckpt = torch.load(checkpoint_path, map_location=device)

    backbone.load_state_dict(unified_ckpt["backbone_state_dict"])
    rpn_head.load_state_dict(unified_ckpt["rpn_state_dict"])
    detection_head.load_state_dict(unified_ckpt["detection_state_dict"])

    rpn_network = RegionProposalNetwork(rpn_head=rpn_head).to(device)
    roi_pool = RoIPool(output_size=(7, 7), pooling_mode="adaptive").to(device)
    detection_network = DetectionNet(detection_head=detection_head).to(device)

    for module in (backbone, rpn_network, roi_pool, detection_network):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    return Pipeline(
        backbone=backbone,
        rpn_network=rpn_network,
        roi_pool=roi_pool,
        detection_network=detection_network,
        device=device,
    )


def predict(pipeline, pil_image, score_thresh, nms_iou_thresh):
    # pil_image: original, un-resized PIL image (RGB). backbone_transform only reads target["size"] (and doesn't need it to hold anything), so an empty dict is enough
    img_tensor, _ = backbone_transform(pil_image, {"size": {}})
    batch_imgs = img_tensor.unsqueeze(0).to(pipeline.device)

    tensor_height, tensor_width = batch_imgs.shape[2], batch_imgs.shape[3]
    img_sizes_before_pad = [(tensor_height, tensor_width)]

    pipeline.detection_network.score_thresh = score_thresh
    pipeline.detection_network.nms_iou_thresh = nms_iou_thresh

    with torch.inference_mode():
        feature_map = pipeline.backbone(batch_imgs)
        _, proposals = pipeline.rpn_network(
            feature_map,
            batch_img_height=tensor_height,
            batch_img_width=tensor_width,
            img_sizes_before_pad=img_sizes_before_pad,
            pre_nms_top_n=6000,
            post_nms_top_n=2000,
        )
        pooled = pipeline.roi_pool(feature_map, proposals, tensor_height, tensor_width)
        labels_list, scores_list, boxes_list = pipeline.detection_network(
            proposals, pooled, img_sizes_before_pad
        )

    labels, scores, boxes = labels_list[0], scores_list[0], boxes_list[0]

    # Rescaling per-axis by the tensor's own dims maps back onto the original image regardless of that swap, since it's the exact inverse of whatever TF.resize did.
    orig_width, orig_height = pil_image.size
    scale_x = orig_width / tensor_width
    scale_y = orig_height / tensor_height

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        rescaled_box = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        detections.append((rescaled_box, VOC_CLASSES[label.item()], score.item()))

    return detections


def draw_boxes(pil_image, detections):
    annotated = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    # Scale font to image size so labels stay legible on both small thumbnails and
    font_size = max(16, round(min(annotated.size) / 40))            # divide by 40 to make font size 2.5% of the smaller image dimension
    font = ImageFont.load_default(size=font_size)

    for (x1, y1, x2, y2), label, score in detections:
        color = _CLASS_COLORS[label]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

        text = f"{label} {score:.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        pad = 2

        # Label goes above the box unless that would run off the top of the image,
        # in which case it's drawn just inside the box instead.
        label_top = y1 - text_height - 2 * pad
        if label_top < 0:
            label_top = y1
        label_bg = (x1, label_top, x1 + text_width + 2 * pad, label_top + text_height + 2 * pad)

        draw.rectangle(label_bg, fill=color)
        draw.text((x1 + pad, label_top + pad - text_bbox[1]), text, fill="white", font=font)

    return annotated
