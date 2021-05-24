import json
import numpy as np
import math
import argparse
import torch
import ConcaveHull as ch
from torch.utils.data import DataLoader
import edge_imageprovider as image_provider
from functools import reduce
import operator
import torch.nn as nn
import sklearn.metrics as sm
import cv2
import torch.nn.functional as F
from skimage.io import imsave



def compute_iou_and_accuracy(arrs, edge_mask1):
    intersection = cv2.bitwise_and(arrs, edge_mask1)
    union = cv2.bitwise_or(arrs, edge_mask1)

    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    iou = intersection_sum / union_sum

    total = np.sum(arrs)
    correct_predictions = intersection_sum

    accuracy = correct_predictions / total
    # print(iou, accuracy)

    return iou, accuracy



    return train_loader

def sort_clockwise(poly):
    coords = poly[:]
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    coords = sorted(coords, key=lambda coord: (-225 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)

    return coords



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


def get_hull(edge_logits):
    test = edge_logits
    test_0 = test[:, :]
    test_1 = test[:, :]

    points_pred = []

    for i in range(len(test_1)):
        for j in range(len(test_1[0])):
            if test_1[i][j] > 0:
                points_pred.append([i, j])

    points_pred = np.asarray(points_pred)

    hull = ch.concaveHull(points_pred, 3)
    # print(hull)
    return hull


def convert_hull_to_cv(hull, w, h):

    original_hull = []
    for i in hull:
        original_hull.append([i[1], i[0]])
    return original_hull

def clockwise_check(points):
    sum = 0
    for i in range(len(points)):
        if i != len(points) - 1:
            sum_x = points[i+1][0] - points[i][0]
            sum_y = points[i+1][1] + points[i][1]
            sum += sum_x * sum_y
        else:
            sum_x = points[0][0] - points[i][0]
            sum_y = points[0][1] + points[i][1]
            sum += sum_x * sum_y
    if sum > 0:
        return True
    else:
        return False





complete_data = []
complete_data1 = []
ptk = 0
d_iou1 = dict()
d_iou_c1 = dict()
d_accuracy1 = dict()
d_accuracy_c1 = dict()

pred_cm = []
gt_cm = []


def testing_hull(poly_logits,class_prob, bbox):
    
    vertex_logits = poly_logits
    # edge_logits = torch.sigmoid(edge_logits)
    # edge_logits = edge_logits[0,0,:,:]
    class_prob = F.log_softmax(class_prob)
    class_prob = torch.squeeze(class_prob)
    # print(class_prob)
    class_label, index = torch.topk(class_prob, 1)
    classes = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']
    

    poly_logits = torch.sigmoid(poly_logits)
    poly_logits = poly_logits[0,0,:,:]

    arrs2 = np.zeros((30, 60), np.uint8)
    for j in range(len(poly_logits)):
        for k in range((len(poly_logits[j]))):
            j1 = math.floor(j)
            k1 = math.floor(k)
            if poly_logits[j][k]>0.51:
                arrs2[j1][k1]= 255


    kernel7 = np.ones((2,2),np.uint8)
    kernel2 = np.ones((3,3),np.uint8)
        # kernel1 = np.ones((2,2),np.uint8)
    # arrs2 = cv2.morphologyEx(arrs2, cv2.MORPH_OPEN, kernel2)
    arrs2 = cv2.morphologyEx(arrs2, cv2.MORPH_CLOSE, kernel2)
    # arrs2 = cv2.morphologyEx(arrs2, cv2.MORPH_CLOSE, kernel7)

    kernel2 = np.ones((3,3),np.uint8)

    borders55 = np.zeros((30, 60), np.float32)

    im, contours, hierarchy = cv2.findContours(arrs2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area=0
    largest_contour=-1
    for i in range(len(contours)):
        cont=contours[i]
        area=cv2.contourArea(cont)
        if(area>max_area):
            max_area=area
            largest_contour=i

    h_contour=contours[largest_contour]
    cv2.polylines(borders55, np.int32([h_contour]),True,[1], thickness = 1)
    

    arrs1 = torch.from_numpy(borders55)
    hull = get_hull(arrs1)

    hull = np.asarray(hull)
    hull = hull.tolist()

    w = bbox[0][2]
    h = bbox[0][3]

    original_hull = convert_hull_to_cv(hull, w, h)
    # original_hull14 = convert_hull_to_cv(hull14, w, h)

    total_points = 100
    original_hull = sort_clockwise(original_hull)
    original_hull = uniformsample(np.asarray(original_hull), total_points).astype(int).tolist()
    

    if clockwise_check(original_hull) == False:
        original_hull = original_hull[::-1]

    if clockwise_check(original_hull) == False:
        print("mismatch!!!!!!!!!!")
    return original_hull