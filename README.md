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


## RoI Pooling

- `RoIPool` projects each proposal (corner format, absolute px) onto the shared feature map using `stride = image_dim // feature_map_dim`, then max-pools each projected region to a fixed 7×7 output — the original Fast R-CNN "RoI Pooling" (quantized max-pool over per-bin `floor`/`ceil` boundaries), not the later RoIAlign (Mask R-CNN's bilinear-interpolated variant).
- Two implementations of the same operation: a manual `"loop"` mode (default) that computes each bin's boundaries explicitly (every bin covers ≥1 pixel even when a proposal is smaller than the output size), and an `"adaptive"` mode via `nn.AdaptiveMaxPool2d`.

## Detection Head (Fast R-CNN)

- Reuses `conv5_x` of an ImageNet-pretrained ResNet-50 as the region classifier — the shared backbone is split at `conv4_x`/`conv5_x`: `conv4_x`'s output is the shared/RPN feature map, `conv5_x` becomes the per-RoI head. BatchNorm affine params are frozen and `.train()` is overridden to keep those BN layers in `eval()` mode (freezing `requires_grad` alone doesn't stop `.train()` from reactivating BN running-stat updates).
- Global average pool over `conv5_x`'s output, then two sibling `nn.Linear` heads:
  - Classification: `num_classes + 1` logits (VOC's 20 object classes + 1 background class).
  - Regression: `num_classes * 4` box deltas — **class-specific**, unlike the RPN's class-agnostic deltas.

## Detection Loss

- Classification: cross-entropy over the sampled proposals (21-way: 20 VOC classes + background).
- Regression: Smooth L1, computed only on the positive proposals' predicted deltas **for their own ground-truth class**, gathered out of the class-specific `[N, num_classes, 4]` delta tensor — summed, then divided by the number of *sampled* proposals for that image (not just the positive count).
- Regression targets are `(t_x, t_y, t_w, t_h)` deltas (same form as the RPN's), normalized by `delta_std = (0.1, 0.1, 0.2, 0.2)` — the Fast R-CNN paper's convention for zero-mean/unit-variance targets. **Unlike the RPN's unnormalized deltas** — any code decoding detection-head deltas back into boxes must multiply by `delta_std` first, or the decoded boxes come out silently near-zero-offset.
- Combined: `loss = cls_loss + λ * reg_loss`, λ = 1 (Fast R-CNN's default balancing weight — unlike the RPN's λ = 10).

## Detection Net (Inference Decode)

- Wraps a trained `DetectionHead` for test-time use: given `RegionProposalNetwork` proposals and their `RoIPool`-ed features, runs the batched detection head once, then per image:
  1. Every `(proposal, foreground class)` pair whose softmax probability exceeds `score_thresh` (default 0.3) is emitted as a candidate — **not** just the argmax class. One proposal can therefore produce several detections, and a proposal whose highest-scoring class is background still contributes its foreground classes.
  2. Each candidate's box deltas for **its own emitted class** are gathered out of the class-specific delta tensor, un-normalized by `delta_std`, and decoded back to boxes with the same center-format inverse transform as the RPN's decoder.
  3. Boxes are clipped to the image's pre-padding size, then boxes that clipping collapsed to under `min_box_size` in either dimension are dropped.
  4. Per-**class** NMS (`torchvision.ops.batched_nms`, IoU `nms_iou_thresh`, default 0.3), then a top-`max_detections_per_image` cap by score (default 100).

### Why argmax was replaced

The original decode kept only the argmax class per proposal and dropped the proposal when that was background. Measured on VOC2007 test, that emitted 16,819 detections against 14,976 GT boxes — 1.12 per object, where a standard Fast R-CNN emits 10–100× more — and capped mean recall at 0.613 while the RPN was supplying 80% proposal recall. Because 11-point AP scores `p_interp(t) = 0` for every `t` above the achieved recall, mAP was pinned at 0.5389 against a ceiling of 0.6046 that the recall alone imposed; precision was already running at 89% of that ceiling. The loss was objects that never became detections at all, not objects ranked badly.

Per-class NMS (rather than class-agnostic) matters for the same metric: a `person` box must not suppress an overlapping `horse` box.

## Training Protocol (4-Step Alternating Training, per the paper)

| Step | What's trained | Backbone |
|---|---|---|
| 1 | RPN (backbone + RPN head, end-to-end) | ImageNet-pretrained, fine-tuned |
| 2 | Fast R-CNN detector, using Step-1 RPN's *frozen* proposals as fixed input | Fresh ImageNet-pretrained, fine-tuned (separate from Step 1's) |
| 3 | RPN again, backbone now frozen (shared, from Step 2) | Frozen |
| 4 | Fast R-CNN unique layers only | Frozen |

**Step 1 hyperparameters** (from the paper): SGD, momentum 0.9, weight decay 0.0005, lr 0.001 for the first ~60k mini-batches then 0.0001 for ~20k more (paper's batch-size-1 framing). This project's realized schedule: batch size 2 (a deliberate deviation for GPU throughput), 10 total epochs over 07+12 (~82,760 iterations) — 8 epochs at lr 0.001, 2 at lr 0.0001.


## Results (VOC2007 test)

### RPN Proposal Recall

| IoU band | Recall | GT boxes recalled |
|---|---|---|
| ≥ 0.5 | 83.91% | 219 / 261 |
| 0.3 – 0.5 | 9.20% | 24 / 261 |
| < 0.3 | 6.90% | 18 / 261 |

### Detection Net (Fast R-CNN head)

| `score_thresh` | `nms_iou_thresh` | mAP @ IoU 0.5 |
|---|---|---|
| 0.1 | 0.3 | **~63%** (best result; per-class AP not separately recorded for this config) |
| 0.3 | 0.3 | 62.52% (full per-class breakdown below) |

Per-class breakdown, `score_thresh=0.3`, `nms_iou_thresh=0.3`:

| Class | AP | rec[-1] | AP_ceil | n_det | n_gt | n_diff |
|---|---|---|---|---|---|---|
| aeroplane | 0.6838 | 0.7333 | 0.7273 | 897 | 285 | 26 |
| bicycle | 0.6937 | 0.7953 | 0.7273 | 852 | 337 | 52 |
| bird | 0.6364 | 0.7211 | 0.7273 | 1359 | 459 | 117 |
| boat | 0.5217 | 0.6540 | 0.6364 | 1951 | 263 | 130 |
| bottle | 0.2293 | 0.3838 | 0.3636 | 1934 | 469 | 188 |
| bus | 0.7971 | 0.9014 | 0.9091 | 869 | 213 | 41 |
| car | 0.6796 | 0.7502 | 0.7273 | 4172 | 1201 | 340 |
| cat | 0.8025 | 0.8966 | 0.8182 | 764 | 358 | 12 |
| chair | 0.3560 | 0.6389 | 0.6364 | 7047 | 756 | 618 |
| cow | 0.6692 | 0.7623 | 0.7273 | 805 | 244 | 85 |
| diningtable | 0.6474 | 0.8447 | 0.8182 | 941 | 206 | 93 |
| dog | 0.7892 | 0.8875 | 0.8182 | 1171 | 489 | 41 |
| horse | 0.7778 | 0.8391 | 0.8182 | 1022 | 348 | 47 |
| motorbike | 0.6966 | 0.7877 | 0.7273 | 890 | 325 | 44 |
| person | 0.5916 | 0.6879 | 0.6364 | 10485 | 4528 | 699 |
| pottedplant | 0.2850 | 0.4729 | 0.4545 | 3228 | 480 | 112 |
| sheep | 0.5681 | 0.6736 | 0.6364 | 880 | 242 | 69 |
| sofa | 0.6941 | 0.8912 | 0.8182 | 2019 | 239 | 157 |
| train | 0.7728 | 0.8688 | 0.8182 | 1163 | 282 | 20 |
| tvmonitor | 0.6119 | 0.7792 | 0.7273 | 2071 | 308 | 53 |

**mAP @ IoU 0.5: 0.6252** — mean recall[-1]: 0.7485 — total detections: 44,520 — total GT (non-difficult): 12,032 — difficult GT excluded: 2,944.

## Known Deviations From the Paper (Summary)

| Deviation | Reason | Expected effect on mAP |
|---|---|---|
| ResNet-50 instead of ResNet-101 | Compute constraint | Weaker features than ResNet-101, likely costing several mAP points — probably felt most on small/textured classes like `bottle`/`pottedplant`, this project's weakest. Not isolated by a ResNet-101 run. |
| Batch size 2 instead of 1 | GPU throughput | Paper's lr schedule (per-image, batch size 1) reused unscaled, changing gradient noise per step. Not isolated. |
| No horizontal flip augmentation | Not implemented | Paper's VOC recipe uses flipping as a free 2× augmentation; skipping it likely costs some mAP, more on sparser classes. Not isolated. |
| BatchNorm frozen from Step 2 onward | Batch size 2 is too small for stable BN statistics — standard practice, not ad hoc | Expected neutral-to-beneficial vs. unfrozen (paper's VGG16 has no BN to compare against). Step 1 is the exception — its backbone trains with BN unfrozen. |

None of these were isolated by a controlled ablation — the ~63% mAP reflects their combined effect, not any single deviation's contribution.


