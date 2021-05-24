import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

EPS = 1e-7

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

class ActiveSplineTorchVaryLength():
    def __init__(self, alpha=0.5, device='cpu'):
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

    def sample_point(self, cps, p_num):
        # Suppose cps is [n_batch, n_cp, 2]
        # p_num is the number to be sampled between two control points
        # print('Sampling points')
        # print(cps.shape)
        # print(p_num)
        cp_num = cps.size(1)
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
        # import ipdb;
        # ipdb.set_trace()
        # No need to calculate gradient w.r.t t.
        t = t.detach()
        # print(t)
        lp = 0
        points = torch.zeros([cps.size(0), p_num * cp_num, cps.size(2)]).float().to(self.device)
        # print(self.device)
        # print(auxillary_cps.type())
        # print(t.type())

        for sg in range(1, cp_num+1):

            # import ipdb;
            # ipdb.set_trace()

            v = self.batch_linspace(t[:, sg], t[:, sg+1], p_num)
            # print(v.type())
            # print(v.size())
            # print(v)
            t0 = t[:, sg-1].unsqueeze(1)
            t1 = t[:, sg].unsqueeze(1)
            t2 = t[:, sg+1].unsqueeze(1)
            t3 = t[:, sg+2].unsqueeze(1)

            for i in range(p_num):

                tv = v[:, i].unsqueeze(1)

                x01 = (t1-tv)/(t1-t0)*auxillary_cps[:, sg-1, :]+(tv-t0)/(t1-t0)*auxillary_cps[:, sg, :]

                x12 = (t2-tv)/(t2-t1)*auxillary_cps[:, sg, :]+(tv-t1)/(t2-t1)*auxillary_cps[:, sg+1,:]

                x23 = (t3-tv)/(t3-t2)*auxillary_cps[:, sg+1, :]+(tv-t2)/(t3-t2)*auxillary_cps[:, sg+2, :]

                x012 = (t2-tv)/(t2-t0)*x01+(tv-t0)/(t2-t0)*x12

                x123 = (t3-tv)/(t3-t1)*x12+(tv-t1)/(t3-t1)*x23

                points[:, lp] = (t2-tv)/(t2-t1)*x012+(tv-t1)/(t2-t1)*x123

                lp = lp + 1

            # print('==> sg: ', sg)
            # if torch.sum(torch.isnan(points)):
            #     print(auxillary_cps)
            #     print(points[0, (sg - 1) * p_num: sg * p_num])
            #     import ipdb;
            #     ipdb.set_trace()
        return points

    def sample_points_gpu_vary(self, pred_polys, masks, cp_p_num):
        pred_polys = pred_polys * masks.unsqueeze(2)
        p_lengths = torch.sum(masks, dim=-1).to(torch.int64)
        sample_points_list = []

        for i in range(p_lengths.shape[0]):
            p = pred_polys[i, :p_lengths[i]]
            # print('==> Before Sample')
            # print(p)
            sample_points = self.sample_point(p.unsqueeze(0), cp_p_num)
            sample_points = torch.clamp(sample_points, 0, 1)
            # print('==> Sample Points')
            # print(sample_points)

            sample_points_list.append(sample_points.squeeze())
        return sample_points_list


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


def plot_spline(xs, ys, x, y, img_name = './image/check_spline.jpg'):
    # plt.axis([-2, 2, -2, 2])
    plt.plot(xs, ys)
    plt.plot(x, y)
    for idx, xy in enumerate(zip(x, y)):
        plt.annotate(str(idx), xy)
    plt.savefig(img_name)
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

def plot_spline_list(cps_list, img_name = './image/check_spline.jpg'):
    # plt.axis([-2, 2, -2, 2])
    np_cps = np.asarray(cps_list)
    plt.plot(np_cps[:,0], np_cps[:,1])
    x = np_cps[:,0]
    y = np_cps[:,1]
    plt.xlim(0,1)
    plt.ylim(0,1)
    # plt.plot(x, y)
    for idx, xy in enumerate(zip(x, y)):
        plt.annotate(str(idx), xy)
    plt.savefig(img_name)
    plt.clf()
    plt.close()

def plot_spline_for_debug():
    diff_iou_init = [[ 0.5941,  0.2273],
         [ 0.5259,  0.2491],
         [ 0.4615,  0.2653],
         [ 0.4013,  0.2829],
         [ 0.3419,  0.3023],
         [ 0.2880,  0.3246],
         [ 0.2397,  0.3515],
         [ 0.1936,  0.3858],
         [ 0.1555,  0.4219],
         [ 0.1190,  0.4587],
         [ 0.0928,  0.5040],
         [ 0.0776,  0.5593],
         [ 0.0803,  0.6121],
         [ 0.0834,  0.6650],
         [ 0.1129,  0.6995],
         [ 0.1561,  0.7142],
         [ 0.1986,  0.7240],
         [ 0.2498,  0.7394],
         [ 0.2965,  0.7628],
         [ 0.3438,  0.7897],
         [ 0.3845,  0.8002],
         [ 0.4148,  0.7910],
         [ 0.4277,  0.7650],
         [ 0.4285,  0.7261],
         [ 0.4377,  0.6759],
         [ 0.4568,  0.6296],
         [ 0.4775,  0.5824],
         [ 0.5074,  0.5493],
         [ 0.5463,  0.5213],
         [ 0.5846,  0.5008],
         [ 0.6281,  0.4814],
         [ 0.6733,  0.4589],
         [ 0.7199,  0.4239],
         [ 0.7594,  0.3873],
         [ 0.7956,  0.3360],
         [ 0.8177,  0.2847],
         [ 0.8170,  0.2395],
         [ 0.7793,  0.2174],
         [ 0.7243,  0.2088],
         [ 0.6615,  0.2132]]

    diff_iou_pred = [[ 0.6266,  0.2211],
         [ 0.5410,  0.2522],
         [ 0.4526,  0.2773],
         [ 0.3693,  0.3049],
         [ 0.2923,  0.3342],
         [ 0.2259,  0.3639],
         [ 0.1850,  0.3979],
         [ 0.1416,  0.4283],
         [ 0.1062,  0.4580],
         [ 0.0831,  0.4861],
         [ 0.0765,  0.5324],
         [ 0.0684,  0.5840],
         [ 0.0677,  0.6432],
         [ 0.0739,  0.6925],
         [ 0.1028,  0.7157],
         [ 0.1418,  0.7153],
         [ 0.1967,  0.6906],
         [ 0.2479,  0.6838],
         [ 0.2951,  0.6981],
         [ 0.3355,  0.7272],
         [ 0.3697,  0.7484],
         [ 0.3973,  0.7527],
         [ 0.4107,  0.7249],
         [ 0.4022,  0.6852],
         [ 0.4179,  0.6424],
         [ 0.4535,  0.6049],
         [ 0.4967,  0.5675],
         [ 0.5484,  0.5389],
         [ 0.6129,  0.5162],
         [ 0.6733,  0.4869],
         [ 0.7371,  0.4635],
         [ 0.7900,  0.4340],
         [ 0.8368,  0.3935],
         [ 0.8629,  0.3531],
         [ 0.8887,  0.2965],
         [ 0.8971,  0.2431],
         [ 0.8799,  0.1949],
         [ 0.8380,  0.1744],
         [ 0.7848,  0.1645],
         [ 0.7175,  0.1834]]

    first_pred =  [[ 0.6733,  0.4589],
         [ 0.7199,  0.4239],
         [ 0.7594,  0.3873],
         [ 0.7956,  0.3360],
         [ 0.8177,  0.2847],
         [ 0.8170,  0.2395],
         [ 0.7793,  0.2174],
         [ 0.7243,  0.2088],
         [ 0.6615,  0.2132],
         [ 0.5941,  0.2273],
         [ 0.5259,  0.2491],
         [ 0.4615,  0.2653],
         [ 0.4013,  0.2829],
         [ 0.3419,  0.3023],
         [ 0.2880,  0.3246],
         [ 0.2397,  0.3515],
         [ 0.1936,  0.3858],
         [ 0.1555,  0.4219],
         [ 0.1190,  0.4587],
         [ 0.0928,  0.5040],
         [ 0.0776,  0.5593],
         [ 0.0803,  0.6121],
         [ 0.0834,  0.6650],
         [ 0.1129,  0.6995],
         [ 0.1561,  0.7142],
         [ 0.1986,  0.7240],
         [ 0.2498,  0.7394],
         [ 0.2965,  0.7628],
         [ 0.3438,  0.7897],
         [ 0.3845,  0.8002],
         [ 0.4148,  0.7910],
         [ 0.4277,  0.7650],
         [ 0.4285,  0.7261],
         [ 0.4377,  0.6759],
         [ 0.4568,  0.6296],
         [ 0.4775,  0.5824],
         [ 0.5074,  0.5493],
         [ 0.5463,  0.5213],
         [ 0.5846,  0.5008],
         [ 0.6281,  0.4814]]

    human_init = [[ 0.0831,  0.4861],
         [ 0.0765,  0.5324],
         [ 0.0684,  0.5840],
         [ 0.0677,  0.6432],
         [ 0.0739,  0.6925],
         [ 0.1028,  0.7157],
         [ 0.1418,  0.7153],
         [ 0.1967,  0.6906],
         [ 0.2479,  0.6838],
         [ 0.2951,  0.6981],
         [ 0.3355,  0.7272],
         [ 0.3697,  0.7484],
         [ 0.3973,  0.7527],
         [ 0.4107,  0.7249],
         [ 0.4022,  0.6852],
         [ 0.4179,  0.6424],
         [ 0.4535,  0.6049],
         [ 0.4967,  0.5675],
         [ 0.5484,  0.5389],
         [ 0.6129,  0.5162],
         [ 0.6733,  0.4869],
         [ 0.7371,  0.4635],
         [ 0.7900,  0.4340],
         [ 0.8368,  0.3935],
         [ 0.8629,  0.3531],
         [ 0.8887,  0.2965],
         [ 0.8971,  0.2431],
         [ 0.8799,  0.1949],
         [ 0.8380,  0.1744],
         [ 0.7848,  0.1645],
         [ 0.7175,  0.1834],
         [ 0.6266,  0.2211],
         [ 0.5410,  0.2522],
         [ 0.4526,  0.2773],
         [ 0.3693,  0.3049],
         [ 0.2923,  0.3342],
         [ 0.2259,  0.3639],
         [ 0.1850,  0.3979],
         [ 0.1416,  0.4283],
         [ 0.1062,  0.4580]]

    human_pred = [[ 0.0714,  0.6817],
         [ 0.0487,  0.7442],
         [ 0.0136,  0.8028],
         [ 0.0118,  0.8586],
         [ 0.0338,  0.8890],
         [ 0.0647,  0.8827],
         [ 0.1194,  0.8504],
         [ 0.1809,  0.7950],
         [ 0.2328,  0.7729],
         [ 0.2765,  0.7743],
         [ 0.3130,  0.8068],
         [ 0.3237,  0.8282],
         [ 0.3459,  0.8272],
         [ 0.3732,  0.8094],
         [ 0.3424,  0.7828],
         [ 0.3442,  0.7559],
         [ 0.3958,  0.7365],
         [ 0.4303,  0.7166],
         [ 0.4715,  0.6972],
         [ 0.5579,  0.6831],
         [ 0.6194,  0.6585],
         [ 0.6523,  0.6183],
         [ 0.7232,  0.5741],
         [ 0.7792,  0.5102],
         [ 0.8027,  0.4455],
         [ 0.8119,  0.3616],
         [ 0.8164,  0.2874],
         [ 0.8127,  0.2226],
         [ 0.7933,  0.1955],
         [ 0.7365,  0.1922],
         [ 0.6811,  0.2401],
         [ 0.6547,  0.3040],
         [ 0.5878,  0.3405],
         [ 0.5159,  0.3843],
         [ 0.4341,  0.4348],
         [ 0.3652,  0.4491],
         [ 0.3035,  0.4719],
         [ 0.2352,  0.5192],
         [ 0.1735,  0.5803],
         [ 0.1131,  0.6335]]

    plot_spline_list(diff_iou_init, './image/diff_iou_init.jpg')
    plot_spline_list(diff_iou_pred, './image/diff_iou_pred.jpg')
    plot_spline_list(first_pred, './image/first_pred.jpg')
    plot_spline_list(human_init, './image/human_init.jpg')
    plot_spline_list(human_pred, './image/human_pred.jpg')


# if __name__ == '__main__':
#     check_spline()
#     # plot_spline_for_debug()
if __name__ == '__main__':
    active_spline = ActiveSpline(3, 10)
    ix = np.asarray([-1, 0, 1, 0, -1])
    iy = np.asarray([0, 1, 0, -2, 0])
    print(ix.shape)
    n = 25
    xs, ys, x, y = active_spline.get_xy(ix, iy, n, 0.5)

    # print(xs, ys)
    plt.axis([-2, 2, -2, 2])
    plt.plot(xs, ys)
    plt.plot(x, y)
    for idx, xy in enumerate(zip(x, y)):
        plt.annotate(str(idx), xy)
    plt.savefig('1.jpg')
    plt.clf()
    plt.close()
    active_spline_torch = ActiveSplineTorch(4, 100, 0.5)
    cps = np.vstack([ix, iy]).T
    cps = cps[:-1]
    print(cps.shape)
    cps = torch.from_numpy(cps).unsqueeze(0).float()
    cps = torch.cat([cps, cps], dim=0)
    cps[1,3,1] = -1
    print(cps)
    points = active_spline_torch.sample_point(cps)
    points = points.numpy()
    cps = cps.numpy()
    idx = 0
    xs = points[idx, :, 0]
    ys = points[idx, :, 1]
    x = cps[idx, :, 0]
    y = cps[idx, :, 1]
    plt.axis([-2, 2, -2, 2])
    plt.plot(xs, ys)
    plt.plot(x, y)
    for idx, xy in enumerate(zip(x, y)):
        plt.annotate(str(idx), xy)
    plt.savefig('1-torch.jpg')
    plt.clf()
    plt.close()
