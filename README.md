# Faster R-CNN From Scratch (PyTorch)

A from-scratch PyTorch implementation of Faster R-CNN (Ren et al., 2015), aiming to closely — not exactly — reproduce paper-level results on Pascal VOC under real compute constraints (Google Colab Free tier, T4 GPU).

## Goal

Follow the original paper's architecture and 4-step alternating training protocol as closely as feasible, while documenting and justifying every deliberate deviation made for compute or practicality reasons.

## Dataset

- **Training**: VOC2007 trainval + VOC2012 trainval ("07+12" protocol), 5,011 + 11,540 = 16,551 images, read from each dataset's `ImageSets/Main/trainval.txt` (not `Segmentation` or `Layout` — an early bug in this project pointed at the wrong subfolder and silently shrank the dataset to ~1,446 images).
- **Evaluation**: VOC2007 test. VOC2012 test is not used, since its ground truth requires official evaluation-server submission.

## Backbone

- **ResNet-50** (deviation from the paper's ResNet-101, for compute reasons — expect a modest mAP gap as a known, accepted trade-off).

- Initialized from ImageNet-pretrained weights.

## Image Preprocessing

- Resize so the shorter side = 600px, longer side capped at 1000px, aspect ratio preserved (matches the paper's protocol; an earlier fixed-224×224 resize was replaced after confirming this in the paper text).
- Ground-truth boxes are denormalized (VOC XML stores them as fractions of original image size) and scaled to match the resized image.
- Since image sizes vary per image under this protocol, **batch size is 1 image**, matching the paper's own "image-centric" mini-batch definition (confirmed directly from Section 3.1.3) — this also avoids the padding/masking machinery that would otherwise be required to batch variable-sized images together.

## Anchors

- 9 anchors per grid location: 3 scales (128², 256², 512²) × 3 ratios (1:1, 1:2, 2:1).
- Represented as `(x_c, y_c, w, h)` throughout generation, encoding, and decoding — converted to corner format `(x1, y1, x2, y2)` where needed (IoU, clipping, NMS).
- Tiled across the RPN's input feature map grid, using `stride = image_dimension // feature_map_dimension` per axis.
- **Known bug (found and fixed)**: `feature_map.shape` unpacking (`N, C, H, W`) was initially reversed (`width, height = shape[2], shape[3]` instead of `height, width = shape[2], shape[3]`), which corrupted anchor y-positions for non-square images. This invalidated an entire completed training run (see Status below) and required anchor generation to be fixed and training restarted.

### Boundary handling (train vs. test)

- **Training**: cross-boundary anchors are excluded entirely from the loss (not labeled positive/negative — ignored).
- **Testing**: no exclusion; decoded proposals are clipped to the image boundary instead.

## Anchor Labeling (IoU-based)

Convention used: positive = `1`, negative = `-1`, ignore = `0`.

- **Positive**: (i) the anchor(s) with the highest IoU for a given GT box, OR (ii) any anchor with IoU > 0.7 with any GT box.
- **Negative**: IoU < 0.3 with all GT boxes.
- **Ignore**: neither of the above — excluded from the loss.
- Condition (i) is applied last, so it can override a negative/ignore label.
- **Design decision**: `matched_gt_idx` is *not* overridden for condition-(i)-only positives (left as the anchor's row-wise best IoU match). This is a deliberate, acknowledged deviation from reference implementations (e.g., torchvision's `Matcher`), traded off against the added complexity of resolving multi-GT-box claims on a single anchor. Documented limitation: in rare cases, a condition-(i) anchor may never receive a regression target pointing at the GT box it was meant to rescue.

## RPN Loss

- 256 anchors sampled per image, ~1:1 positive:negative ratio (padded with negatives if fewer than 128 positives are available).
- Classification: cross-entropy over the sampled anchors (2-class: background/foreground).
- Regression: Smooth L1 on `(t_x, t_y, t_w, t_h)` deltas, positive anchors only, normalized by positive count.
- Combined: `loss = cls_loss + λ * reg_loss`, λ = 10.
- Box regression targets/predictions are always in `t`-space (`t_x, t_y, t_w, t_h`) — the network never predicts absolute coordinates directly; `encode_deltas`/`decode_box_deltas` convert between GT/anchor pairs and this space.

## Proposal Generation (decode → clip → filter → NMS → top-N)

No explicit pre-NMS top-N truncation is used (the original paper doesn't specify one; this differs from later reference implementations like torchvision/Detectron2, which add one purely for speed).

1. Decode anchors + predicted deltas → boxes
2. Clip to image boundary (test-time; training excludes cross-boundary anchors upstream instead)
3. Filter boxes smaller than 16px
4. NMS, IoU threshold 0.7
5. Keep top-N: **2000** proposals during training (fed to the eventual Fast R-CNN stage), **300** during inference (paper's main reported test-time setting; the paper's own ablations also try 100/1000/6000).

## Training Protocol (4-Step Alternating Training, per the paper)

| Step | What's trained | Backbone |
|---|---|---|
| 1 | RPN (backbone + RPN head, end-to-end) | ImageNet-pretrained, fine-tuned |
| 2 | Fast R-CNN detector, using Step-1 RPN's *frozen* proposals as fixed input | Fresh ImageNet-pretrained, fine-tuned (separate from Step 1's) |
| 3 | RPN again, backbone now frozen (shared, from Step 2) | Frozen |
| 4 | Fast R-CNN unique layers only | Frozen |

**Step 1 hyperparameters** (from the paper): SGD, momentum 0.9, weight decay 0.0005, lr 0.001 for the first ~60k mini-batches then 0.0001 for ~20k more (paper's batch-size-1 framing). This project's realized schedule: batch size 2 (a deliberate deviation for GPU throughput), 10 total epochs over 07+12 (~82,760 iterations) — 8 epochs at lr 0.001, 2 at lr 0.0001.

**Critical operational reminder**: after loading `optimizer_state_dict` from a checkpoint, the learning rate must be explicitly reset via `param_groups` — the old LR otherwise persists silently across a resume, even if a new LR is passed to the optimizer's constructor.

## Current Status

- RPN architecture (head, loss, proposal generation) implemented.
- Step 1 training run **completed once** (10 epochs, batch size 2) but is **invalid** due to the anchor-generation shape-unpacking bug described above — visual inspection and IoU checks on the trained model's proposals showed systematically incorrect box coordinates (`y_min` pinned near 0 for most top proposals), traced back to corrupted anchor y-positions during training.
- Anchor generation bug fixed and verified (anchor coordinate ranges now consistent with actual feature map / image dimensions).
- **Full Step 1 retraining is required** from ImageNet-pretrained weights, using the corrected anchor generation throughout (both label assignment and delta encoding depend on it).
- Recommended before re-committing to a full 10-epoch run: a small-scale overfit test (4–8 fixed images, several hundred iterations) to confirm the corrected pipeline can actually drive IoU up on a controlled set, before spending another ~6+ hours of Colab compute on the full run.

## Known Deviations From the Paper (Summary)

| Deviation | Reason |
|---|---|
| ResNet-50 instead of ResNet-101 | Compute constraint |
| Batch size 2 instead of 1 | GPU throughput on T4 |
| No explicit pre-NMS top-N | Paper doesn't specify one |
| `matched_gt_idx` not overridden for condition-(i) positives | Simplicity; acknowledged edge-case limitation |

## Environment

- Google Colab Free tier (T4 GPU, ~12GB VRAM, ~12-hour session cap)
- Checkpoints and logs persisted to Google Drive; dataset unzipped to local Colab storage for I/O speed
- PyTorch, torchvision (`ops.nms`, ResNet-50 pretrained weights)

## Not Yet Started

- ROI Pooling / ROI Align
- Fast R-CNN detection head (classification into 21 VOC classes + box regression)
- Fast R-CNN loss and proposal-to-GT matching/sampling
- Steps 2–4 of alternating training
- Final inference pipeline (per-class NMS, confidence thresholding)
- mAP evaluation (VOC's 11-point interpolated AP protocol)
