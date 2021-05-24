import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import argparse
import torch
from tooloscrptgcn import Model
import ConcaveHull as ch
from torch.utils.data import DataLoader
# import operator
import torch.nn as nn
from collections import OrderedDict
from simplification.cutil import simplify_coords
from skimage.io import imsave, imread
from test_hull5 import testing_hull
from skimage.transform import rescale, resize
# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
import utils
import visualize
from visualize import display_images
# import mrcnn_st.model as modellib
# from mrcnn_st.model import log

# from mrcnn_st.model import log


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("!!!!!!got cuda!!!!!!!")
else:
    print("!!!!!!!!!!!!no cuda!!!!!!!!!!!!")

def uniformsample_batch(batch, num):
    final = []
    for i in batch:
        i1 = np.asarray(i).astype(int)
        a = uniformsample(i1, num)
        a = torch.from_numpy(a)
        a = a.long()
        final.append(a)
    return final

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

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', type=str)
    parser.add_argument('--bbox', type=str, default=None)

    args = parser.parse_args()

    return args

def get_hull(edge_logits):
    test = edge_logits
    test_0 = test[:, :]
    test_1 = test[:, :]

    


    # for i in range(len(test_0)):
    #     for j in range(len(test_0[0])):
    #         if test_0[i][j] > 0.7:
    #             test_1[i][j] = 1
    #         else:
    #             test_1[i][j] = 0
    points_pred = []

    for i in range(len(test_1)):
        for j in range(len(test_1[0])):
            if test_1[i][j] > 0:
                points_pred.append([i + 1, j + 1])

    points_pred = np.asarray(points_pred)

    hull = ch.concaveHull(points_pred, 3)
    return hull

def convert_hull_to_cv(hull, bbox):

    original_hull = []

    w = bbox[2]
    h = bbox[3]

    for i in hull:
        original_hull.append([int((i[1]) * w / 60), int((i[0]) * h / 30)])
    return original_hull

# model_path = './checkpoints_cgcn/Final.pth'
model = Model().to(device)
# if torch.cuda.device_count() >= 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [20, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = nn.DataParallel(model)
state_dict = torch.load('./checkpoints_cgcn/GCNbeforeFT.pth')
new_state_dict = OrderedDict()
for k, v in state_dict["gcn_state_dict"].items():
    name = k[7:] # remove `module.`
    # print(k,v)
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
model.to(device)

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

f = open('masks_with_Indiscapes.json',"r")
data = json.load(f)

def get_edge(i_url,bbox):

    imgs = []
    bboxs1 = []
    # print(bbox)
    # if len(bbox) < 1:
    #     # print(bbox)
    #     converted_data = {"image_url": i_url, "bboxs": bbox, "masks":[]}
    # else:
    for i in range(len(bbox)):
        print(i_url)
        img_orig = cv2.imread(i_url)
        # print(img.shape)
        w_img = img_orig.shape[1]
        h_img = img_orig.shape[0]

        # img = cv2.resize(img, (1024, 1024))

        # print(bbox[i])
        x0 = int(bbox[i][0])
        y0 = int(bbox[i][1])
        w = int(bbox[i][2])
        h = int(bbox[i][3])


        img1 = img_orig[y0:y0+h,x0:x0+w]

        # img= cv2.copyMakeBorder(img,0,0,10,10,cv2.BORDER_REPLICATE)
        img1 = cv2.resize(img1, (960, 1920))
        # bboxs1.append()
        # print(img)
        imgs.append(img1)

    imgs = np.asarray(imgs)
    # print(imgs.shape)
    # print(imgs)
    img = torch.from_numpy(imgs)
    # print(img.shape)
    # img = torch.cat(img)
    img = img.view(-1, 960, 1920, 3)
    img = torch.transpose(img, 1, 3)
    img = torch.transpose(img, 2, 3)
    img = img.float()
    dp = torch.zeros((len(i_url),1000,2), dtype = torch.float32)
    output_dict, poly_logits, class_prob = model(img.to(device), bbox, dp)
    pred_cps = output_dict['pred_polys'][-1]
    # pred_cps = pred_cps.cpu().detach().numpy()
    # pred_cps = pred_cps.tolist()

    final_mask = []
    label = []
    for i in range(len(bbox)):
        x0 = int(bbox[i][0])
        y0 = int(bbox[i][1])
        w = int(bbox[i][2])
        h = int(bbox[i][3])
        # pred_cps1 = convert_hull_to_cv(pred_cps[i], bbox[i])
        pred_cps1 = pred_cps[i]
        # print(class_prob[i])
        # class_prob = torch.log_softmax(class_prob,1)
        # print(class_prob)
        # print(indi)
        indi = torch.argmax(class_prob,1)
        # print(indi[0])
        # class_prob = torch.zeros((1,8), dtype = torch.float32)
        # class_label = indi[0].cpu().detach().numpy()
        # class_label = class_label[0]
        
        pred_x = ((pred_cps1[:, 0] * h)+y0).view(1000,1)
        pred_y = ((pred_cps1[:, 1] * w)+x0).view(1000,1)

        pred = torch.cat((pred_y, pred_x), dim=1)
        # print(pred)
        # pred_cps1 = uniformsample(np.asarray(pred_cps1),200)
        # pred_cps1 = simplify_coords(pred_cps1, 0.1)
        pred = pred.cpu().detach().numpy()
        pred = np.asarray(pred)
        
        mask = np.zeros((h_img, w_img))
        cv2.fillPoly(mask, np.int32([pred]), [1])
        # imsave("./test_gcn_pred6/" + str(i) + "mask_palm_leaf.jpg", mask)
        # mask = mask.gt(0.5)
        mask = mask > 0.5
        final_mask.append(mask)
        # label.append(class_label)

    final_mask = np.asarray(final_mask)
    # label = np.asarray(label)
    final_mask = np.moveaxis(final_mask,0,-1)
    ax = get_ax(1)
	# r = results[0]
	# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
	#                             dataset.class_names, r['scores'], ax=ax,
	#                             title="Predictions")
	# log("gt_class_id", gt_class_id)
	# log("gt_bbox", gt_bbox)
	# log("gt_mask", gt_mask)
    r = {}
    rois = bbox
    masks = final_mask
    class_ids = [2,2,2,2]
    class_names = ['Hole(Physical)', 'Character Line Segment', 'Physical Degradation',
				 'Character Component', 'Picture', 'Decorator', 'Library Marker', 'Boundary Line']
	#print(r)
    ccc,contours=visualize.display_instances(img_orig, r['rois'], r['masks'], r['class_ids'], 
        dataset.class_names, None, ax=ax,
        title="Predictions",show_bbox=False,show_mask=True)

		

if __name__ == '__main__':
    # args = get_args()
    # img = args.img
    # bbox1 = args.bbox
    # bbox1 = bbox1.split(",")
    # x, y = get_edge("./Andros332.jpg", [41,240,909,168])
    image_list = ["./54_2.jpg"]
    # bbox = [[41,240,909,168]]
    complete_data = []
    f = open('Abhi.json',"r")
    data = json.load(f)
    # print(data['_via_img_metadata'][0])
    data = data["_via_img_metadata"]
    keys = data.keys()
    keys_list = [k for k in keys]
    idx = 0
    key = 0
    bboxs = []
    for k in keys_list:
        tmp = data[k]
        regions = tmp["regions"]
        # print(regions)
        for r in regions:
            if(r==None):
                continue
            # count+=1
            # dict = {}
            x_cor = []
            y_cor = []
            # area = 0.0
            poly =[]
            r_keys = [k for k in r.keys()]
            
            if("shape_attributes" not in r_keys):
                continue
            shape_attributes = r["shape_attributes"]
            attributes = [x for x in shape_attributes.keys()]
            if("all_points_x" in attributes):
                x_cor = shape_attributes["all_points_x"]
                y_cor = shape_attributes["all_points_y"]
            else:
                bbox = [shape_attributes["x"],shape_attributes["y"],shape_attributes["width"],shape_attributes["height"]]
            bboxs.append(bbox)
    # print(data[0])
    print(bboxs)
    ct = 0
    for i in range(len(image_list)):
        ct += 1
        # if ct > 5:
        #     continue
        i_url = image_list[i]
        # df1 = img.replace("%20", " ")
        # n = 3
        # start = df1.find("/")
        # while start >= 0 and n > 1:
        #     start = df1.find("/", start + len("/"))
        #     n -= 1

        # df2 = df1[start:]
        # i_url = "./new_jpg_data"+df2
        # print(i_url)
        # bbox = bboxs[i]
        get_edge(i_url, bboxs)    
        # complete_data.append(conv_data)
