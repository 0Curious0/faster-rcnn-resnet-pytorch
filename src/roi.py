import torch
import torch.nn as nn

class RoIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, feature_maps, batch_proposals, batch_img_height, batch_img_width):
        channels = feature_maps.shape[1]
        feat_height, feat_width = feature_maps.shape[2], feature_maps.shape[3]
        out_h, out_w = self.output_size

        stride_x = batch_img_width // feat_width
        stride_y = batch_img_height // feat_height

        pooled_batch = []
        for feature_map, proposals in zip(feature_maps, batch_proposals):
            if proposals.shape[0] == 0:
                # No proposals survived for this image — keep a well-shaped empty tensor
                pooled_batch.append(feature_map.new_empty((0, channels, out_h, out_w)))
                continue

            projected = self._project_to_feature_map(proposals, feat_height, feat_width, stride_x, stride_y)

            pooled_proposals = []
            for proposal_coords in projected:
                roi_feature_map = self._select_roi_feature_map(feature_map, proposal_coords)
                pooled_proposals.append(self._max_pool_roi(roi_feature_map))

            pooled_batch.append(torch.stack(pooled_proposals, dim=0))  # [N_i, C, out_h, out_w]

        return pooled_batch

    def _project_to_feature_map(self, proposals, feat_height, feat_width, stride_x, stride_y):
        x1 = proposals[:, 0]
        y1 = proposals[:, 1]
        x2 = proposals[:, 2]
        y2 = proposals[:, 3]

        fx1 = torch.round(x1 / stride_x)
        fy1 = torch.round(y1 / stride_y)
        fx2 = torch.round(x2 / stride_x)
        fy2 = torch.round(y2 / stride_y)

        # Convert to long type for holding larger values and prevent overflow due to decimal double
        fx1 = fx1.long()
        fy1 = fy1.long()
        fx2 = fx2.long()
        fy2 = fy2.long()

        fx1 = torch.clamp(fx1, 0, feat_width - 1)
        fx2 = torch.clamp(fx2, 0, feat_width - 1)
        fy1 = torch.clamp(fy1, 0, feat_height - 1)
        fy2 = torch.clamp(fy2, 0, feat_height - 1)

        return torch.stack([fx1, fy1, fx2, fy2], dim=-1)  # [N, 4]

    def _select_roi_feature_map(self, feature_map, proposal_coords):
        fx1, fy1, fx2, fy2 = proposal_coords

        return feature_map[:, fy1:fy2 + 1, fx1:fx2 + 1]  # [C, roi_h, roi_w]

    def _max_pool_roi(self, roi_feature_map):
        channels, roi_h, roi_w = roi_feature_map.shape
        out_h, out_w = self.output_size  # output_size is (H, W) of the pooled map
        device = roi_feature_map.device

        # Bin boundaries per axis (each axis uses its own bin count): floor for
        # start, ceil for end so every bin covers >= 1 pixel even when the ROI
        # is smaller than the number of bins along that axis.
        h_idx = torch.arange(out_h, device=device)
        h_starts = torch.clamp(torch.floor(h_idx * roi_h / out_h).long(), 0, roi_h)
        h_ends = torch.clamp(torch.ceil((h_idx + 1) * roi_h / out_h).long(), 0, roi_h)

        w_idx = torch.arange(out_w, device=device)
        w_starts = torch.clamp(torch.floor(w_idx * roi_w / out_w).long(), 0, roi_w)
        w_ends = torch.clamp(torch.ceil((w_idx + 1) * roi_w / out_w).long(), 0, roi_w)

        pooled = roi_feature_map.new_empty((channels, out_h, out_w))
        for ph in range(out_h):
            for pw in range(out_w):
                bin_region = roi_feature_map[:, h_starts[ph]:h_ends[ph], w_starts[pw]:w_ends[pw]]
                pooled[:, ph, pw] = bin_region.amax(dim=(1, 2))  # max over the bin, per channel

        return pooled  # [C, out_h, out_w]
