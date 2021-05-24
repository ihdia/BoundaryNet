import torch
from torch import nn
import numpy as np
import render_shape as rs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_line_params(x0, y0, x1, y1, norm=False):
    """
    Return A,B,C where the line from (x0,y0) to
    (x1,y1) is Ax + By + C = 0

    set norm=True to get the unit vector of the
    normal
    """
    A = y1 - y0
    B = -(x1 - x0)
    C = x1 * y0 - y1 * x0

    if norm:
        n = (A ** 2 + B ** 2) ** 0.5
        return A, B, C, n
    else:
        return A, B, C


def get_closest_point(x0, y0, a, b, c):
    """
    Returns closest point from x0,y0 to
    ax + by + c = 0
    """

    x = (b * (b * x0 - a * y0) - a * c) / (a ** 2 + b ** 2)
    y = (a * (-b * x0 + a * y0) - b * c) / (a ** 2 + b ** 2)

    return x, y


class DiffIoU(nn.Module):
    def __init__(self, dim=100):
        super(DiffIoU, self).__init__()
        self.dim = dim
        self.sample_ds = 0.5
        self.max_samples = int(self.dim / self.sample_ds) + 1
        self.sample_ds_arr = torch.arange(self.max_samples).to(device)

    def polygon_area(self, poly):
        """
        poly: [bs, num_v, 2]
        """
        y_max, _ = torch.max(poly[:, :, 1], dim=-1)

        num_v = poly.size(1)
        for i in torch.arange(num_v, dtype=torch.int64):
            next_i = (i + 1) % num_v
            x0, x1, y0, y1 = poly[:, i, 0], poly[:, next_i, 0], poly[:, i, 1], poly[:, next_i, 1]

            area_line = (x1 - x0) * (y_max - (y1 + y0) / 2)

            if i == 0:
                s = area_line
            else:
                s += area_line

        return torch.abs(s)

    def get_uniques(self, samples):
        floored = torch.floor(samples).to(torch.int32)

        floored[1:] = floored[1:] - floored[:-1]
        floored[0] = 1  # make sure 1st index is selected

        idxs = torch.nonzero(floored)
        idxs = idxs.view(-1)

        return idxs

    def get_samples(self, x0, x1, y0, y1,
                    direction='x', masked=True):
        v_y = y1 - y0 + 1e-6
        v_x = x1 - x0 + 1e-6
        norm = (v_y ** 2 + v_x ** 2) ** 0.5
        v_x = v_x / norm
        v_y = v_y / norm

        x_s = x0 + self.sample_ds_arr * v_x
        y_s = y0 + self.sample_ds_arr * v_y

        # Mask out those that get out of range
        x_mask = (x_s <= torch.max(x0, x1) + 1e-3) & (x_s >= torch.min(x0, x1) - 1e-3)
        y_mask = (y_s <= torch.max(y0, y1) + 1e-3) & (y_s >= torch.min(y0, y1) - 1e-3)
        mask = x_mask & y_mask

        x_s = torch.masked_select(x_s, mask)
        y_s = torch.masked_select(y_s, mask)

        if masked:
            if direction == 'x':
                bound_mask = (x_s <= self.dim - 1) & (x_s >= 0.0)
            else:
                bound_mask = (y_s <= self.dim - 1) & (y_s >= 0.0)

            x_s = torch.masked_select(x_s, bound_mask)
            y_s = torch.masked_select(y_s, bound_mask)

            if direction == 'x':
                uniq_samples = x_s
            else:
                uniq_samples = y_s

            if uniq_samples.size(0) > 0:
                idxs = self.get_uniques(uniq_samples)

                x_s = x_s[idxs]
                y_s = y_s[idxs]

        return x_s, y_s

    def line_sum_mask(self, mask, x0, x1, y0, y1, direction):
        out = torch.zeros(mask.size(0)).to(device)

        for i in range(mask.size(0)):
            x_s, y_s = self.get_samples(x0[i], x1[i], y0[i], y1[i], direction, masked=True)
            s = self.interpolated_sum(mask[i], x_s, y_s)

            out[i] = torch.sum(s)

        return out

    def interpolated_sum(self, mask, Xs, Ys):
        """
        Find bilinear interpolated value in mask for each
        (x,y) in zip(x_s,y_s) and sum
        """
        X0 = torch.floor(Xs)
        X1 = X0 + 1
        Y0 = torch.floor(Ys)
        Y1 = Y0 + 1

        w_00 = (X1 - Xs) * (Y1 - Ys)
        w_01 = (X1 - Xs) * (Ys - Y0)
        w_10 = (Xs - X0) * (Y1 - Ys)
        w_11 = (Xs - X0) * (Ys - Y0)

        # clip to range [0, dim-1] to not violate boundaries
        X0 = X0.clamp(0, self.dim - 1).to(torch.int64)
        X1 = X1.clamp(0, self.dim - 1).to(torch.int64)
        Y0 = Y0.clamp(0, self.dim - 1).to(torch.int64)
        Y1 = Y1.clamp(0, self.dim - 1).to(torch.int64)

        M_00 = mask[Y0, X0]
        M_01 = mask[Y1, X0]
        M_10 = mask[Y0, X1]
        M_11 = mask[Y1, X1]

        return w_00 * M_00 + w_01 * M_01 + w_10 * M_10 + w_11 * M_11

    def polygon_area_mask(self, poly, mask, direction):
        """
        poly: [bs, num_v, 2]
        mask: [bs, num_mask, dim, dim, ]
        direction: [num_mask, ]
        """
        num_v = poly.size(1)

        for m in torch.arange(mask.size(1), dtype=torch.int64):
            for i in torch.arange(num_v, dtype=torch.int64):
                next_i = (i + 1) % num_v
                x0, x1, y0, y1 = poly[:, i, 0], poly[:, next_i, 0], poly[:, i, 1], poly[:, next_i, 1]

                if direction[m] == 'x':
                    sign = 2 * ((x1 > x0).float() - 0.5)
                else:
                    sign = 2 * ((y1 > y0).float() - 0.5)

                sum_f = self.line_sum_mask(mask[:, m], x0, x1, y0, y1, direction[m])
                sum_b = self.line_sum_mask(mask[:, m], x1, x0, y1, y0, direction[m])

                if i == 0:
                    s = sign * ((sum_f + sum_b) / 2)
                else:
                    s += sign * ((sum_f + sum_b) / 2)

            if m == 0:
                ret = torch.abs(s)
            else:
                ret += torch.abs(s)

        return ret / mask.size(1)

    def get_iou(self, poly, gt, gt_mask, direction):
        pred_area = self.polygon_area(poly)
        gt_area = self.polygon_area(gt)

        int_area = self.polygon_area_mask(poly, gt_mask, direction)

        union_area = pred_area + gt_area - int_area

        return int_area / union_area

    def sample_dist(self, s1, s2):
        s1 = torch.stack(s1, dim=-1)  # (M,2)
        s2 = torch.stack(s2, dim=-1)  # (N,2)

        s1 = s1 / self.dim
        s2 = s2 / self.dim
        # Normalize coordinates

        s1 = torch.unsqueeze(s1, dim=-2)  # (M, 1, 2)
        s2 = torch.unsqueeze(s2, dim=-3)  # (1, N, 2)

        s1 = s1.repeat(1, s2.size(-2), 1)  # (M, N, 2)

        d = (s1 - s2) ** 2
        d = torch.sum(d, dim=-1)  # (M, N)

        d = 1 / (d + 1e-8)

        d = torch.sum(d)

        return d

    def self_intersection_iben(self, poly):
        """
        Self intersection energy from:
        http://graphics.berkeley.edu/papers/Iben-RPP-2009-04/Iben-RPP-2009-04.pdf

        poly: [bs, num_v, 2]
        """
        poly = poly / self.dim
        poly2 = poly.detach()
        num_v = poly.size(1)

        cross_terms = torch.zeros(poly.size(0)).to(device)
        self_terms = torch.zeros(poly.size(0)).to(device)

        for i in range(num_v):
            for j in range(num_v):
                if j == i or j == (i - 1) % num_v:
                    continue

                e = (poly2[:, j, :], poly2[:, (j + 1) % num_v, :])
                d0 = torch.sum((e[0] - poly[:, i, :]) ** 2)
                d1 = torch.sum((e[1] - poly[:, i, :]) ** 2)

                a, b, c = get_line_params(e[0][:, 0], e[0][:, 1], e[1][:, 0], e[1][:, 1])
                d2 = ((a * poly[:, i, 0] + b * poly[:, i, 1] + c) ** 2) / (a ** 2 + b ** 2)

                t = ((poly[:, i, 0] - e[0][:, 0]) * (e[1][:, 0] - e[0][:, 0]) \
                     + (poly[:, i, 1] - e[0][:, 1]) * (e[1][:, 1] - e[0][:, 1])) \
                    / (1e-6 + (e[1][:, 0] - e[0][:, 0]) ** 2 + (e[1][:, 1] - e[0][:, 1]) ** 2)

                d = (t <= 0).float() * d0 + (t >= 1).float() * d1 + \
                    ((t > 0) & (t < 1)).float() * d2

                cross_terms += 1 / (d + 1e-6)

        for i in range(num_v):
            e = (poly[:, i, :], poly[:, (i + 1) % num_v, :])
            mod_e = 1e-6 + torch.sum((e[1] - e[0]) ** 2, dim=-1)
            s = mod_e + (torch.log(mod_e) - 1)
            self_terms += s

        return cross_terms + self_terms

    def self_intersection_edge_samples(self, poly):
        """
        Self intersection energy from:
        https://pdfs.semanticscholar.org/0ce5/13f6a440f7b68a7e048efb94655109843c26.pdf

        poly: [bs, num_v, 2]
        """
        num_v = poly.size(1)

        cross_terms = torch.zeros(poly.size(0)).to(device)
        self_terms = torch.zeros(poly.size(0)).to(device)

        for b in range(poly.size(0)):
            fwd_samples = {}
            bwd_samples = {}

            for i in range(num_v - 2):
                e0 = (poly[b, i, :], poly[b, (i + 1) % num_v, :])
                for j in range((i + 2) % num_v, num_v - int(i == 0)):
                    e1 = (poly[b, j, :], poly[b, (j + 1) % num_v, :])
                    if i not in fwd_samples.keys():
                        fwd_samples[i] = self.get_samples(e0[0][0], e0[1][0], e0[0][1], e0[1][1], masked=False)
                        bwd_samples[i] = self.get_samples(e0[1][0], e0[0][0], e0[1][1], e0[0][1], masked=False)
                    if j not in fwd_samples.keys():
                        fwd_samples[j] = self.get_samples(e1[0][0], e1[1][0], e1[0][1], e1[1][1], masked=False)
                        bwd_samples[j] = self.get_samples(e1[1][0], e1[0][0], e1[1][1], e1[0][1], masked=False)

                    s = self.sample_dist(fwd_samples[i], fwd_samples[j])
                    s = 0.5 * (s + self.sample_dist(bwd_samples[i], bwd_samples[j]))

                    s = s / self.sample_ds
                    # Normalize by number of samples

                    cross_terms[b] += s

        for i in range(num_v):
            e = (poly[:, i, :], poly[:, (i + 1) % num_v, :])
            mod_e = 1e-6 + torch.sum((e[1] - e[0]) ** 2, dim=-1)
            s = mod_e + (torch.log(mod_e) - 1)
            self_terms += s

        return cross_terms + self_terms

    def forward(self, poly, gt):
        """
        poly must be a tensor with requires_grad = True

        poly: [bs, num_v, 2]
        gt: [bs, num_v_gt, 2]
        """
        with torch.no_grad():
            gt_c = gt.cpu().numpy()
            gt_mask = []
            for i in range(gt.shape[0]):
                gt_mask_list = rs.create_shape_mask([self.dim, self.dim], gt_c[i, ...],
                                                    bi_d=True, add_x=True)

                (a, b), (c, d) = gt_mask_list
                gt_mask.append(np.stack([a, b, c, d], axis=0))

            gt_mask = torch.from_numpy(np.array(gt_mask))
            direction = ['x', 'x', 'y', 'y']

        iou = self.get_iou(
            poly,
            gt,
            gt_mask.to(device),
            direction
        )

        return iou

    def dt_loss(self, poly, gt):
        """
        Distance Transform Loss
        """
        gt_dt = []
        with torch.no_grad():
            gt_c = gt.cpu().numpy()
            for i in range(gt.size(0)):
                gt_dt.append(rs.dt_image([self.dim, self.dim], gt_c[i]))

        gt_dt = torch.from_numpy(np.array(gt_dt)).to(device)

        Xs = poly[:, :, 0]
        Ys = poly[:, :, 1]

        # Clip extra distances
        Xs_clip = torch.clamp(Xs, 0, self.dim - 1)
        Ys_clip = torch.clamp(Ys, 0, self.dim - 1)
        extra_dist = torch.abs(Xs - Xs_clip) + torch.abs(Ys - Ys_clip)

        normal_loss = torch.zeros(poly.size(0)).to(device)
        for i in range(poly.size(0)):
            normal_loss[i] = torch.sum(self.interpolated_sum(gt_dt[i], Xs[i], Ys[i]))

        loss = torch.sum(normal_loss) + torch.sum(extra_dist)

        return loss

    def vertex_affinity(self, poly, gt):
        """
        poly: [bs, M, 2]
        gt: [bs, N, 2]
        """
        poly = torch.unsqueeze(poly, dim=-2)  # (bs, N, 1, 2)
        gt = torch.unsqueeze(gt, dim=-3)  # (bs, 1, M, 2)

        gt = gt.repeat(1, poly.size(1), 1, 1)  # (bs, N, M, 2)

        dist = (gt - poly) ** 2

        dist = torch.sum(dist, dim=-1) ** 0.5  # (bs, N, M)

        dist, _ = torch.min(dist, dim=-1)  # (bs, N)

        dist = torch.sum(dist, dim=-1)  # (bs)

        return dist

    def chamfer_loss(self, poly, gt):
        """
        Chamfer loss, symmetric
        """
        poly = 1e-6 + poly / self.dim
        gt = gt / self.dim

        loss = self.vertex_affinity(poly, gt) + self.vertex_affinity(gt, poly)

        return loss