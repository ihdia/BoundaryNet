import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import active_spline.render_shape as rs
import time

def plt_img(img_array, name):
    plt.imshow(img_array)
    plt.savefig(name)
    plt.close

def vis_image(pred_points, save_name):
    fig, ax = plt.subplots()
    # ax.plot(gt_points[:,0], gt_points[:,1], '-o')
    ax.plot(pred_points[:, 0], pred_points[:, 1], '-o')
    ax.set_aspect('equal')
    # ax.set_origin('upper')
    plt.gca().invert_yaxis()
    plt.savefig(save_name, origin='upper')
    plt.close()
    # plt.clf()

def plot_sum_same(img, s_value, x, y, save_name):

    fig, ax = plt.subplots()
    ax.imshow(img)
    # ax.plot(gt_points[:,0], gt_points[:,1], '-o')
    # ax.plot(pred_points[:, 0], pred_points[:, 1], '-o')
    ax.set_aspect('equal')
    ax.scatter(x,y, c=s_value)
    plt.savefig(save_name, origin='upper')
    plt.close()

def plot_sum(s_value, x, y, save_name):
    fig, ax = plt.subplots()
    # ax.plot(gt_points[:,0], gt_points[:,1], '-o')
    # ax.plot(pred_points[:, 0], pred_points[:, 1], '-o')
    ax.set_aspect('equal')
    ax.scatter(x,y, c=s_value)
    plt.savefig(save_name, origin='upper')
    plt.close()
    # plt.clf()
    # pass

EPS = 1e-7

class DiffIOU(nn.Module):
    def __init__(self, dim, device):
        super(DiffIOU, self).__init__()
        self.dim = dim
        self.device = device

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

    def get_gt_mask(self, gt):
        with torch.no_grad():
            gt_c = gt.cpu().numpy()
            gt_mask = []
            # print(gt.shape)
            for i in range(gt.shape[0]):
                gt_mask_list = rs.create_shape_mask([self.dim, self.dim], gt_c[i, ...],
                                                    bi_d=True, add_x=True)

                (a, b), (c, d) = gt_mask_list
                gt_mask.append(np.stack([a, b, c, d], axis=0))

            gt_mask = torch.from_numpy(np.array(gt_mask)).float().to(self.device)
            direction = ['x', 'x', 'y', 'y']
            return gt_mask, direction

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


    def get_unique_mask(self, points, dim):
        # Remove the points in the same pixel
        # print(points)
        floored = torch.floor(points).to(torch.float)
        next_points = points[:, 1:, :]
        next_points = torch.cat([next_points, points[:, 0, :].unsqueeze(1)], dim=1)
        next_floor = torch.floor(next_points).to(torch.float)
        floor_x = next_floor[:,:,0] - floored[:,:,0]
        floor_x[:,0] = 1
        floor_y = next_floor[:, :, 1] - floored[:, :, 1]
        floor_y[:,0] = 1
        floor_x = torch.abs(floor_x)
        floor_y = torch.abs(floor_y)
        floor_x[floor_x>1] = 1
        floor_y[floor_y>1] = 1
        return floor_x, floor_y


    def get_xsign(self, points):
        next_points = points[:, 1:, :]
        next_points = torch.cat([next_points, points[:, 0, :].unsqueeze(1)], dim=1)
        diff = next_points[:, :, 0] - points[:, :, 0]
        return torch.sign(diff)
        # return 1

    def get_ysign(self, points):
        # Get the sign for points acumulate in y axes
        # Points: [n_batch, n_points, 2]
        next_points = points[:,1:,:]
        next_points = torch.cat([next_points, points[:,0,:].unsqueeze(1)], dim=1)
        diff = next_points[:,:,1] - points[:,:,1]
        return torch.sign(diff)


    def polygon_area_mask(self, pred_points, gt_mask, direction):

        x_mask, y_mask = self.get_unique_mask(pred_points, self.dim)

        x_sign = self.get_xsign(pred_points)
        y_sign = self.get_ysign(pred_points)


        ret = torch.zeros(pred_points.size(0)).float().to(self.device)
        for i in range(gt_mask.size(0)):
            for m in torch.arange(gt_mask.size(1), dtype=torch.int64):

                s = self.interpolated_sum(gt_mask[i,m], pred_points[i,:,0],   pred_points[i,:,1])

                if direction[m] == 'y':
                    s = y_sign[i] * s
                    s = y_mask[i] * s
                else:
                    s = x_sign[i] * s
                    s = x_mask[i] * s

                s = torch.sum(s)
                ret[i] = ret[i] + torch.abs(s)

        return ret / gt_mask.size(1)

    def get_iou(self, pred_points, gt_points):
        # print('==> Get individual areas')
        print(pred_points.shape)
        print(gt_points.shape)
        pred_points = pred_points * self.dim
        gt_points = gt_points * self.dim

        pred_area = self.polygon_area(pred_points)
        gt_area = self.polygon_area(gt_points)

        # print('Ratio: ', pred_area / gt_area)

        gt_mask, direction = self.get_gt_mask(gt_points)

        # for m in range(gt_mask.size(1)):
        #     mask = gt_mask[0,m]
        #     plt_img(mask, 'tmp/%d.jpg'%(m))

        int_area = self.polygon_area_mask(pred_points, gt_mask, direction)
        # print('==> Intersection ', int_area)
        # print(pred_area.type())
        # print(gt_area.type())
        # print(int_area.type())
        union_area = pred_area + gt_area - int_area
        # print('==> Ground Area ',  gt_area, pred_area)
        # print('==> Union ', union_area)
        return torch.mean(int_area / union_area)







class ActiveSplineTorch():
    def __init__(self, cp_num, p_num, alpha=0.5, device='cpu'):
        self.cp_num = cp_num
        self.p_num = int(p_num / cp_num)
        self.alpha = alpha
        self.device = device

    def batch_arange(self, start_t, end_t, step_t):
        batch_arr = map(torch.arange, start_t, end_t, step_t)
        batch_arr = [arr.unsqueeze(0) for arr in batch_arr]
        return torch.cat(batch_arr, dim=0)

    def batch_linspace(self, start_t, end_t, step_t):
        step_t = [step_t] * end_t.size(0)
        batch_arr = map(torch.linspace, start_t, end_t, step_t)
        batch_arr = [arr.unsqueeze(0) for arr in batch_arr]
        return torch.cat(batch_arr, dim=0).to(self.device)

    def sample_point(self, cps):
        # Suppose cps is [n_batch, n_cp, 2]
        cps = torch.cat([cps, cps[:, 0, :].unsqueeze(1)], dim=1)
        auxillary_cps = torch.zeros(cps.size(0), cps.size(1) + 2, cps.size(2)).to(self.device)
        auxillary_cps[:, 1:-1, :] = cps

        l_01 = torch.sqrt(torch.sum(torch.pow(cps[:, 0, :] - cps[:, 1, :], 2), dim=1) + EPS)
        l_last_01 = torch.sqrt(torch.sum(torch.pow(cps[:, -1, :] - cps[:, -2, :], 2), dim=1) + EPS)

        l_01.detach_().unsqueeze_(1)
        l_last_01.detach_().unsqueeze_(1)

        # print(l_last_01, l_01)

        auxillary_cps[:, 0, :] = cps[:, 0, :] - l_01 / l_last_01 * (cps[:, -1, :] - cps[:, -2, :])
        auxillary_cps[:, -1, :] = cps[:, -1, :] + l_last_01 / l_01 * (cps[:, 1, :] - cps[:, 0, :])

        # print(auxillary_cps)

        t = torch.zeros([auxillary_cps.size(0), auxillary_cps.size(1)]).to(self.device)

        for i in range(1, t.size(1)):
            t[:, i] = torch.pow(torch.sqrt(torch.sum(torch.pow(auxillary_cps[:, i, :] - auxillary_cps[:, i-1, :], 2),
                                                    dim=1)), self.alpha) + t[:, i-1]

        # No need to calculate gradient w.r.t t.
        t = t.detach()
        # print(t)
        lp = 0
        points = torch.zeros([cps.size(0), self.p_num * self.cp_num, cps.size(2)]).to(self.device)
        # print(self.device)
        # print(auxillary_cps.type())
        # print(t.type())

        for sg in range(1, self.cp_num+1):
            v = self.batch_linspace(t[:, sg], t[:, sg+1], self.p_num)
            # print(v.type())
            # print(v.size())
            # print(v)
            t0 = t[:, sg-1].unsqueeze(1)
            t1 = t[:, sg].unsqueeze(1)
            t2 = t[:, sg+1].unsqueeze(1)
            t3 = t[:, sg+2].unsqueeze(1)

            for i in range(self.p_num):

                tv = v[:, i].unsqueeze(1)

                x01 = (t1-tv)/(t1-t0)*auxillary_cps[:, sg-1, :]+(tv-t0)/(t1-t0)*auxillary_cps[:, sg, :]

                x12 = (t2-tv)/(t2-t1)*auxillary_cps[:, sg, :]+(tv-t1)/(t2-t1)*auxillary_cps[:, sg+1,:]

                x23 = (t3-tv)/(t3-t2)*auxillary_cps[:, sg+1, :]+(tv-t2)/(t3-t2)*auxillary_cps[:, sg+2, :]

                x012 = (t2-tv)/(t2-t0)*x01+(tv-t0)/(t2-t0)*x12

                x123 = (t3-tv)/(t3-t1)*x12+(tv-t1)/(t3-t1)*x23

                points[:, lp] = (t2-tv)/(t2-t1)*x012+(tv-t1)/(t2-t1)*x123

                lp = lp + 1

        return points


class ActiveSpline():
    def __init__(self, cp_num, p_num):
        self.cp_num = cp_num
        self.p_num = p_num

    def get_xy(self, ix, iy, n, al):
        x = np.zeros([ix.shape[0] + 2])
        y = np.zeros([iy.shape[0] + 2])
        x[1:-1] = ix
        y[1:-1] = iy
        l_01 = np.sqrt(np.square(ix[0]-ix[1]) + np.square(iy[0]-iy[1]))
        l_last_01 = np.sqrt(np.square(ix[-1]-ix[-2]) + np.square(iy[-1]-iy[-2]))
        print(l_last_01, l_01)
        x[0] = ix[0] - l_01 / l_last_01 * (ix[-1] - ix[-2])
        y[0] = iy[0] - l_01 / l_last_01 * (iy[-1] - iy[-2])
        x[-1] = ix[-1] + l_last_01 / l_01 * (ix[1] - ix[0])
        y[-1] = iy[-1] + l_last_01 / l_01 * (iy[1] - iy[0])

        # print(x, y)
        n_cp = x.shape[0]
        n_acp = n_cp - 2
        t = np.zeros([x.shape[0]])

        for i in range(1, t.shape[0]):
            t[i] = np.power(np.sqrt(np.power(x[i]-x[i-1], 2)+np.power(y[i]-y[i-1], 2)), al) + t[i-1]

        # print(t)

        lp = 0
        xs = np.zeros([n*(n_acp-1)])
        ys = np.zeros([n*(n_acp-1)])

        for sg in range(1, n_acp):
            v = np.arange(t[sg], t[sg+1], (t[sg+1]-t[sg])/n)
            print(v)
            t0 = t[sg-1]
            t1 = t[sg]
            t2 = t[sg+1]
            t3 = t[sg+2]

            if sg == n_acp:
                n = n + 1

            for i in range(0, n):
                tv = v[i]

                x01 = (t1-tv)/(t1-t0)*x[sg-1]+(tv-t0)/(t1-t0)*x[sg]
                y01 = (t1-tv)/(t1-t0)*y[sg-1]+(tv-t0)/(t1-t0)*y[sg]

                x12 = (t2-tv)/(t2-t1)*x[sg]+(tv-t1)/(t2-t1)*x[sg+1]
                y12 = (t2-tv)/(t2-t1)*y[sg]+(tv-t1)/(t2-t1)*y[sg+1]

                x23 = (t3-tv)/(t3-t2)*x[sg+1]+(tv-t2)/(t3-t2)*x[sg+2]
                y23 = (t3-tv)/(t3-t2)*y[sg+1]+(tv-t2)/(t3-t2)*y[sg+2]

                x012 = (t2-tv)/(t2-t0)*x01+(tv-t0)/(t2-t0)*x12
                y012 = (t2-tv)/(t2-t0)*y01+(tv-t0)/(t2-t0)*y12

                x123 = (t3-tv)/(t3-t1)*x12+(tv-t1)/(t3-t1)*x23
                y123 = (t3-tv)/(t3-t1)*y12+(tv-t1)/(t3-t1)*y23

                xs[lp] = (t2-tv)/(t2-t1)*x012+(tv-t1)/(t2-t1)*x123
                ys[lp] = (t2-tv)/(t2-t1)*y012+(tv-t1)/(t2-t1)*y123

                lp = lp + 1
        return xs, ys, x, y


def plot_spline(xs, ys, x, y):
    # plt.axis([-2, 2, -2, 2])
    plt.plot(xs, ys)
    plt.plot(x, y)
    for idx, xy in enumerate(zip(x, y)):
        plt.annotate(str(idx), xy)
    plt.savefig('check_spline.jpg')
    plt.clf()
    plt.close()


def check_spline():
    import json
    root_dir = '/ais/gobi6/jungao/polyrnn-pp-chamfer/model_dirs/active-spline-20-cp/prediction/epoch13_step_34000'
    import glob
    import os
    file_list = glob.glob(os.path.join(root_dir, '*.json'))
    print(len(file_list))
    file_name = []
    for name in file_list:
        # print(name)
        if 'frankfurt_000000_011810_43' in name:
            # print(name)
            file_name = name

    ann = json.load(open(file_name, 'r'))
    spline_pos = ann['spline_pos']
    spline_pos = [np.array(spline_pos)]
    active_spline = ActiveSpline(20, 250)
    xs, ys, x, y = active_spline.get_xy(spline_pos[0][:,0], spline_pos[0][:, 1], 250, 0.5)
    plot_spline(xs, ys, x, y)

    # instance = json.load(open())


if __name__ == '__main__':
    check_spline()

