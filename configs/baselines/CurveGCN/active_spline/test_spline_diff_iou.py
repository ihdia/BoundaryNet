from ActiveSplineDiffIOU import ActiveSplineTorch, DiffIOU
import random
random.seed(1)
import os
from glob import glob
import shutil
import json
import numpy as np
import torch
import skimage.io as io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def compute_iou_cityscapes(pred, gt):
    """
    Compute IOU [Cityscapes mode].

    Args:
        pred: pred mask.
        gt: gt mask.
    """
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    pred_area = np.count_nonzero(pred)
    gt_area = np.count_nonzero(gt)
    print('Ground Truth Area : %d, Predict Area : %d, ratio: %.5f'%(gt_area, pred_area, float(pred_area) / float(gt_area)))

    # true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))

    union = float(true_positives + false_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union if union > 0. else 0.

    return iou

def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i];

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp

def copy_json_for_test():
    root_path = '/ais/gobi6/jungao/polyrnn-pp-chamfer/model_dirs/active-spline-20-cp/prediction/epoch13_step_34000'
    change_path = '/ais/gobi6/jungao/polyrnn-pp-chamfer/model_dirs/active-spline-20-cp/prediction/test_diff_iou'
    print('==> List dir')
    anns = glob(os.path.join(root_path, '*.json'))
    random.shuffle(anns)
    if not os.path.exists(change_path):
        os.makedirs(change_path)

    print('==> Copy')
    for i in range(100):
        print(i)
        shutil.copy(anns[i], change_path)

def vis_image(gt_points, pred_points, save_name):
    fig, ax = plt.subplots()
    ax.plot(gt_points[:,0], gt_points[:,1], '-o')
    ax.plot(pred_points[:, 0], pred_points[:, 1], '-o')
    ax.set_aspect('equal')
    # ax.set_origin('upper')
    plt.gca().invert_yaxis()
    plt.savefig(save_name, origin='upper')

def get_real_iou(ann, preds_dir):
    gt_mask = io.imread(os.path.join(preds_dir, ann['gt_mask_fname']))
    pred_mask = io.imread(os.path.join(preds_dir, ann['pred_mask_fname']))

    # Get IOU
    iou = compute_iou_cityscapes(pred_mask, gt_mask)
    return iou


def check_json_spline(file_name, preds_dir):
    print('==> Get Positions')
    ann= json.load(open(file_name, 'r'))
    spline_pos = ann['spline_pos']
    spline_pos = np.array(spline_pos)
    original_polys = [np.array(comp['poly']) for comp in ann['components']][0]
    print(spline_pos.shape)
    print(original_polys.shape)

    print('==> Get real IOU')
    real_iou = get_real_iou(ann, preds_dir)
    print(real_iou)

    # vis_image(original_polys, spline_pos, 'original_img.jpg')
    print('==> Normalize Positions')
    original_polys = original_polys.astype(np.float)
    x_min = np.min(original_polys[:,0]) - 10
    y_min = np.min(original_polys[:,1]) - 10
    x_max = np.max(original_polys[:,0]) + 10
    y_max = np.max(original_polys[:,1]) + 10
    scale = max(x_max - x_min, y_max - y_min)
    spline_pos[:,0] = (spline_pos[:,0] - x_min) / scale
    spline_pos[:,1] = (spline_pos[:,1] - y_min) / scale
    original_polys[:,0] = (original_polys[:,0] - x_min) / scale
    original_polys[:,1] = (original_polys[:,1] - y_min) / scale

    # print(spline_pos)
    # print(original_polys)
    # vis_image(original_polys, spline_pos, 'normalize_img.jpg')

    print('==> Sample Points from the curve and line')
    active_spline = ActiveSplineTorch(20, 1000)
    s_ps = active_spline.sample_point(torch.from_numpy(spline_pos).unsqueeze(0))
    # p_ps = uniformsample(original_polys, 1000)
    p_ps = original_polys
    p_ps = torch.from_numpy(p_ps).unsqueeze(0).expand(2, p_ps.shape[0],2).float()
    s_ps = s_ps.expand(2, 1000, 2).float()
    # vis_image(p_ps[0].numpy(), s_ps[0].numpy(), 'sampled_points.jpg')

    print('==> match for iou')
    d_iou = DiffIOU(224)
    iou = d_iou.get_iou(s_ps, p_ps)

    print('Real IOU: %.4f, Pred IOU: %.4f'%(real_iou, iou[0]))
    # print(iou)



def check_all_json():
    root_path = '/ais/gobi6/jungao/polyrnn-pp-chamfer/model_dirs/active-spline-20-cp/prediction/test_diff_iou'
    file_list = glob(os.path.join(root_path, '*.json'))
    preds_path = '/ais/gobi6/jungao/polyrnn-pp-chamfer/model_dirs/active-spline-20-cp/prediction/epoch13_step_34000'
    # file_name = 'frankfurt_000001_010830_39_info.json'
    # file_name = os.path.join(root_path, file_name)
    for file_name in file_list:
        # print(file_name)
        print(file_name)
        # file_name = '/ais/gobi6/jungao/polyrnn-pp-chamfer/model_dirs/active-spline-20-cp/prediction/test_diff_iou/frankfurt_000000_017228_72_info.json'
        # file_name = '/ais/gobi6/jungao/polyrnn-pp-chamfer/model_dirs/active-spline-20-cp/prediction/test_diff_iou/frankfurt_000001_017459_64_info.json'
        check_json_spline(file_name, preds_path)
        # exit()

if __name__ == '__main__':
    # copy_json_for_test()
    # check_json_spline()
    check_all_json()