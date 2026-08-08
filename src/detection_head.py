import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class DetectionHead(nn.Module):
    def __init__(self, num_classes=20, pretrained=True):
        super().__init__()

        self.num_classes = num_classes

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.conv5_x = resnet50(weights=weights).layer4

        self.fc_cls = nn.Linear(2048, num_classes + 1)
        self.fc_reg = nn.Linear(2048, num_classes * 4)

        self.fc_cls.weight.data.normal_(0, 0.01)
        self.fc_reg.weight.data.normal_(0, 0.001)
        self.fc_cls.bias.data.zero_()
        self.fc_reg.bias.data.zero_()

        for module in self.conv5_x.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def train(self, mode=True):
            super().train(mode)
    
            # requires_grad=False only freezes the affine params
            # training mode may again put the batchnorm layers in training mode, which is not what we want
            for module in self.conv5_x.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
    
            return self

    def forward(self, pooled_batch):
        # pooled_batch: list of B tensors [N_i, 1024, 7, 7] from RoIPool
        counts = [pooled.shape[0] for pooled in pooled_batch]

        # No proposals survived anywhere in the batch -- torch.cat would still work, but
        # conv5_x on a 0-length batch is pointless, so return correctly shaped empties
        if sum(counts) == 0:
            cls_logits = [pooled.new_zeros((0, self.num_classes + 1)) for pooled in pooled_batch]
            box_deltas = [pooled.new_zeros((0, self.num_classes * 4)) for pooled in pooled_batch]
            return cls_logits, box_deltas

        # One batched pass over every RoI in the batch, then split back per image
        x = torch.cat(pooled_batch, dim=0)           # [sumN, 1024, 7, 7]
        x = self.conv5_x(x)                          # [sumN, 2048, 4, 4]
        x = x.mean(dim=(2, 3))                       # global average pool -> [sumN, 2048]

        cls_logits = self.fc_cls(x)                  # [sumN, 21] raw logits
        box_deltas = self.fc_reg(x)                  # [sumN, 80] class-specific deltas

        return list(cls_logits.split(counts)), list(box_deltas.split(counts))


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_maps, batch_proposals, batch_gt_boxes, batch_gt_labels,
                batch_img_height, batch_img_width, roi_pool, detection_head,
                num_samples=64, pos_fraction=0.25, background_label=20):
        # feature_maps:   [B, 1024, H_f, W_f] from Backbone
        # batch_proposals: list of B x [N_i, 4] CORNER absolute px, from RegionProposalNetwork
        # batch_gt_boxes:  list of B x [M_i, 4] CORNER absolute px
        # batch_gt_labels: list of B x [M_i] in 0..19
        device = feature_maps.device
        batch_size = len(batch_proposals)

        batch_sampled_proposals = []
        batch_labels = []
        batch_reg_targets = []

        # 64 sampled RoIs per image instead of all ~2000 proposals.
        for i in range(batch_size):
            # Step 2 trains the detector on the step-1 RPN's proposals as FIXED input.
            # reg_targets are built from these boxes, so without the detach the regression
            # target itself would be differentiable and smooth_l1_loss would backpropagate
            # into the RPN.
            proposals = batch_proposals[i].detach()
            gt_boxes = batch_gt_boxes[i].to(device)
            gt_labels = batch_gt_labels[i].to(device)

            labels, matched_gt_idx = self.assign_proposal_labels(proposals, gt_boxes, gt_labels, background_label=background_label)
            sampled_idx = self.sample_proposals(labels, num_samples, pos_fraction, background_label)

            sampled_proposals = proposals[sampled_idx]
            sampled_labels = labels[sampled_idx]

            if gt_boxes.shape[0] == 0:
                # Every label is background here, so these targets are never read.
                # Indexing gt_boxes with matched_gt_idx would be out of bounds.
                reg_targets = torch.zeros((sampled_idx.numel(), 4), dtype=sampled_proposals.dtype, device=device)   # unused placeholder gets discarded by selecting only foreground labels
            else:
                matched_gt_boxes = gt_boxes[matched_gt_idx[sampled_idx]]
                reg_targets = self.encode_box_targets(sampled_proposals, matched_gt_boxes)

            batch_sampled_proposals.append(sampled_proposals)
            batch_labels.append(sampled_labels)
            batch_reg_targets.append(reg_targets)

        pooled_batch = roi_pool(feature_maps, batch_sampled_proposals, batch_img_height, batch_img_width)
        batch_cls_logits, batch_box_deltas = detection_head(pooled_batch)

        total_cls_loss = 0.0
        total_reg_loss = 0.0

        for i in range(batch_size):
            cls_logits = batch_cls_logits[i]
            box_deltas = batch_box_deltas[i]
            labels = batch_labels[i]
            reg_targets = batch_reg_targets[i]

            num_sampled = labels.numel()
            if num_sampled == 0:
                # Nothing survived sampling for this image; it contributes zero to both terms
                continue

            cls_loss = nn.functional.cross_entropy(cls_logits, labels, reduction='mean')

            # For reg_loss, selecting sampled proposals that are not background (i.e., positive samples) is necessary because the regression loss is only computed for foreground classes.
            positive_mask = labels != background_label

            if positive_mask.sum() == 0:
                # smooth_l1_loss on an empty tensor returns nan and would poison the batch
                reg_loss = torch.tensor(0.0, dtype=box_deltas.dtype, device=device)
            else:
                # box_deltas is [N, 4 * 20] 
                num_classes = box_deltas.shape[1] // 4
                positive_labels = labels[positive_mask]
                positive_deltas = box_deltas[positive_mask].view(-1, num_classes, 4)
                # Select the predicted deltas for each proposal based on proposal idx and proposal_label(for that prop_idx)
                # torch.arange(positive_labels.numel(), device=device) creates [0, ..., P = num_positive_proposal], positive_labels = [cls_label_prop_1, cls_label_prop_2, ..., cls_label_prop_P]
                predicted_deltas = positive_deltas[torch.arange(positive_labels.numel(), device=device), positive_labels]

                reg_loss = nn.functional.smooth_l1_loss(predicted_deltas, reg_targets[positive_mask], reduction='sum')

                reg_loss = reg_loss / num_sampled

            total_cls_loss += cls_loss
            total_reg_loss += reg_loss

        return total_cls_loss / batch_size, total_reg_loss / batch_size

    def compute_iou_matrix_corners(self, boxes_a, boxes_b):
        a_x1, a_y1, a_x2, a_y2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
        b_x1, b_y1, b_x2, b_y2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

        inter_x1 = torch.max(a_x1.unsqueeze(1), b_x1.unsqueeze(0))
        inter_y1 = torch.max(a_y1.unsqueeze(1), b_y1.unsqueeze(0))
        inter_x2 = torch.min(a_x2.unsqueeze(1), b_x2.unsqueeze(0))
        inter_y2 = torch.min(a_y2.unsqueeze(1), b_y2.unsqueeze(0))

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
        area_b = (b_x2 - b_x1) * (b_y2 - b_y1)

        union_area = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter_area

        return inter_area / union_area

    def assign_proposal_labels(self, proposals, gt_boxes, gt_labels, pos_iou_thresh=0.5, neg_iou_lo=0.1, background_label=20):
        # proposals: [N, 4] CORNER absolute px, straight from RegionProposalNetwork
        # gt_boxes:  [M, 4] CORNER absolute px    gt_labels: [M] in 0..19
        # Returns labels [N] (0..19 foreground, 20 background, -1 ignore) and
        # matched_gt_idx [N] (index of the best-overlapping gt box for each proposal).
        num_proposals = proposals.shape[0]

        if gt_boxes.shape[0] == 0:
            # No objects in this image, so every proposal is background
            labels = torch.full((num_proposals,), background_label, dtype=torch.long, device=proposals.device)
            matched_gt_idx = torch.zeros((num_proposals,), dtype=torch.long, device=proposals.device)  # unused placeholder
            return labels, matched_gt_idx

        iou_matrix = self.compute_iou_matrix_corners(proposals, gt_boxes)
        max_iou_per_proposal, matched_gt_idx = iou_matrix.max(dim=1)

        labels = torch.full((num_proposals,), -1, dtype=torch.long, device=proposals.device)
        labels[max_iou_per_proposal >= neg_iou_lo] = background_label

        positive_mask = max_iou_per_proposal >= pos_iou_thresh
        labels[positive_mask] = gt_labels[matched_gt_idx[positive_mask]]

        return labels, matched_gt_idx

    def sample_proposals(self, labels, num_samples=64, pos_fraction=0.25, background_label=20):
        # 64 per image, 25% of them positive.
        positive_idx = torch.where((labels >= 0) & (labels != background_label))[0]
        negative_idx = torch.where(labels == background_label)[0]

        num_pos = min(int(num_samples * pos_fraction), positive_idx.numel())
        num_neg = min(num_samples - num_pos, negative_idx.numel())

        perm_pos = torch.randperm(positive_idx.numel(), device=labels.device)[:num_pos]
        perm_neg = torch.randperm(negative_idx.numel(), device=labels.device)[:num_neg]

        sampled_pos_idx = positive_idx[perm_pos]
        sampled_neg_idx = negative_idx[perm_neg]

        # positives first, then negatives -- ignored proposals never enter the result
        return torch.cat([sampled_pos_idx, sampled_neg_idx])

    def corners_to_center(self, boxes):
        # (x1, y1, x2, y2) -> (x_c, y_c, w, h)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        x_c = x1 + w / 2
        y_c = y1 + h / 2
        return torch.stack([x_c, y_c, w, h], dim=1)

    def encode_box_targets(self, proposals, gt_boxes, delta_std=(0.1, 0.1, 0.2, 0.2)):
        proposals_centered = self.corners_to_center(proposals)
        gt_centered = self.corners_to_center(gt_boxes)

        p_xc, p_yc, p_w, p_h = proposals_centered[:, 0], proposals_centered[:, 1], proposals_centered[:, 2], proposals_centered[:, 3]
        gt_xc, gt_yc, gt_w, gt_h = gt_centered[:, 0], gt_centered[:, 1], gt_centered[:, 2], gt_centered[:, 3]

        target_dx = (gt_xc - p_xc) / p_w
        target_dy = (gt_yc - p_yc) / p_h
        target_dw = torch.log(gt_w / p_w)
        target_dh = torch.log(gt_h / p_h)

        target_deltas = torch.stack((target_dx, target_dy, target_dw, target_dh), dim=1)

        # Standard Deviations of the regression targets
        std = torch.tensor(delta_std, dtype=target_deltas.dtype, device=target_deltas.device)

        # Based on the Dataset, the mean of target_deltas is appx 0 and std dev (0.1, 0.1, 0.2, 0.2)
        # So, by (target_deltas - 0)/std we are normalizing the regression targets to have zero mean and unit variance, as described in the Fast R-CNN paper.
        return target_deltas / std
