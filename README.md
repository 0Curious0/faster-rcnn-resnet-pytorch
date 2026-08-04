# Faster R-CNN From Scratch (PyTorch)

A from-scratch PyTorch implementation of Faster R-CNN (Ren et al., 2015), aiming to closely — not exactly — reproduce paper-level results on Pascal VOC under real compute constraints.

## Dataset

- **Training**: VOC2007 trainval + VOC2012 trainval ("07+12" protocol), 5,011 + 11,540 = 16,551 images, read from each dataset's `ImageSets/Main/trainval.txt` (not `Segmentation` or `Layout` — an early bug in this project pointed at the wrong subfolder and silently shrank the dataset to ~1,446 images).
- **Evaluation**: VOC2007 test. VOC2012 test is not used, since its ground truth requires official evaluation-server submission.

## Backbone

- **ResNet-50** (deviation from the paper's ResNet-101, for compute reasons — expect a modest mAP gap as a known, accepted trade-off).

- Initialized from ImageNet-pretrained weights.

## Image Preprocessing

- Resize so the shorter side = 600px, longer side capped at 1000px, aspect ratio preserved (matches the paper's protocol; an earlier fixed-224×224 resize was replaced after confirming this in the paper text).
- Images are padded with max_dimensions in the batch to create a batch image tensor

## Anchors

- 9 anchors per grid location: 3 scales (128², 256², 512²) × 3 ratios (1:1, 1:2, 2:1).

### Boundary handling (train vs. test)

- **Training**: cross-boundary anchors are excluded entirely from the loss (not labeled positive/negative — ignored).
- **Testing**: no exclusion; decoded proposals are clipped to the image boundary instead.

## Anchor Labeling (IoU-based)

Convention used: positive = `1`, negative = `-1`, ignore = `0`.

- **Positive**: (i) the anchor(s) with the highest IoU for a given GT box, OR (ii) any anchor with IoU > 0.7 with any GT box.
- **Negative**: IoU < 0.3 with all GT boxes.
- **Ignore**: neither of the above — excluded from the loss.
- Condition (i) is applied last, so it can override a negative/ignore label.


## RPN Loss

- 256 anchors sampled per image, ~1:1 positive:negative ratio (padded with negatives if fewer than 128 positives are available).
- Classification: cross-entropy over the sampled anchors (2-class: background/foreground).
- Regression: Smooth L1 on `(t_x, t_y, t_w, t_h)` deltas, positive anchors only, normalized by positive count.
- Combined: `loss = cls_loss + λ * reg_loss`, λ = 10.

## Proposal Generation (decode → clip → filter → NMS → top-N)


1. Decode anchors + predicted deltas → boxes
2. Clip to image boundary (test-time; training excludes cross-boundary anchors upstream instead)
3. Filter boxes smaller than 16px
4. Select Pre-NMS-Top-N Boxes
4. NMS, IoU threshold 0.7
5. Keep Post-NMS-Top-N Boxes


## Training Protocol (4-Step Alternating Training, per the paper)

| Step | What's trained | Backbone |
|---|---|---|
| 1 | RPN (backbone + RPN head, end-to-end) | ImageNet-pretrained, fine-tuned |
| 2 | Fast R-CNN detector, using Step-1 RPN's *frozen* proposals as fixed input | Fresh ImageNet-pretrained, fine-tuned (separate from Step 1's) |
| 3 | RPN again, backbone now frozen (shared, from Step 2) | Frozen |
| 4 | Fast R-CNN unique layers only | Frozen |

**Step 1 hyperparameters** (from the paper): SGD, momentum 0.9, weight decay 0.0005, lr 0.001 for the first ~60k mini-batches then 0.0001 for ~20k more (paper's batch-size-1 framing). This project's realized schedule: batch size 2 (a deliberate deviation for GPU throughput), 10 total epochs over 07+12 (~82,760 iterations) — 8 epochs at lr 0.001, 2 at lr 0.0001.


## Known Deviations From the Paper (Summary)

| Deviation | Reason |
|---|---|
| ResNet-50 instead of ResNet-101 | Compute constraint |
| Batch size 2 instead of 1 |


