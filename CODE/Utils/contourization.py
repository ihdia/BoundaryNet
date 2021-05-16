import json
import numpy as np
import math
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from scipy.interpolate import splprep, splev
import Utils.utils
from skimage.transform import rescale, resize
from skimage import img_as_bool
import skimage

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

def testing_hull(poly_logits,class_prob, bbox):
    try:
        class_prob = F.softmax(class_prob)
        class_prob = torch.squeeze(class_prob)
        # print(class_prob)
        class_label, index = torch.topk(class_prob, 1)
        classes = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']
        
        pred_label =classes[index[0]]

        w1,h1 = poly_logits[:,0,:,:].shape[1],poly_logits[:,0,:,:].shape[2]

        poly_logits88 = torch.sigmoid(poly_logits[0,0,:,:]).detach().cpu().numpy()
        yy = poly_logits88 > 0.5
        yy = yy+0
        poly_logits88 = yy.astype(np.float32)

        arrs2 = (poly_logits88*255).astype(np.uint8)

        
        kernel2 = np.ones((3,3),np.uint8)
        arrs2 = cv2.morphologyEx(arrs2, cv2.MORPH_CLOSE, kernel2)


        w = bbox[0][2]
        h = bbox[0][3]

        if w != h1 or h != w1:
            arrs2 = (arrs2/255).astype(np.int)
            arrs2 = (arrs2).astype(np.bool)
            arrs2 = resize(arrs2,(h, w))
            arrs2 = img_as_bool(arrs2)
            arrs2 = arrs2 + 0
            # print(arrs2)
            arrs2 = (arrs2*255).astype(np.uint8)


        contours, hierarchy = cv2.findContours(arrs2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        max_area=0
        largest_contour=-1
        cont_list = []
        for i in range(len(contours)):
            cont=contours[i]
            area=cv2.contourArea(cont)
            if(area>max_area):
                max_area=area
                largest_contour=i
            cont_list.append(i)
        
        cont_list1 = []
        x1 = []
        y1 = []
        list_p = []
        for i in cont_list:
            if cv2.contourArea(contours[i]) > max_area/(7):
                cont_list1.append(i)
                M = cv2.moments(contours[i])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x1.append(cx)
                y1.append(cy)
                list_p.append([cx,cy])

        if len(list_p) > 1:
            bfr = arrs2
            bfr = skimage.color.gray2rgb(bfr)
            for i in list_p:
                cv2.circle(bfr, (int(i[0]), int(i[1])), 7, (210, 0, 0), -1)

            borders55 = np.zeros((h,w), np.float32)
            cv2.polylines(borders55, np.int32([list_p]),True,[1], thickness = int(h/7))


            arrs2 = (arrs2/255).astype(np.int)
            borders55 = borders55.astype(np.int)
            arrs2 = np.bitwise_or(arrs2,borders55)

            arrs2 = (arrs2*255).astype(np.uint8)

            contours1, hierarchy1 = cv2.findContours(arrs2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            max_area=0
            largest_contour1=-1
            cont_list = []
            for i in range(len(contours1)):
                cont=contours1[i]
                area=cv2.contourArea(cont)
                if(area>max_area):
                    max_area=area
                    largest_contour1=i

            try:
                h_contour = contours1[largest_contour1]
            except:
                h_contour = np.zeros((5,2))
                h_contour[:,1] = [1,w1-2,w1-2,1,1]
                h_contour[:,0] = [1,1,h1-2,h1-2,1]
        else:
            try:
                h_contour = contours[largest_contour]
            except:
                h_contour = np.zeros((5,2))
                h_contour[:,1] = [1,w1-2,w1-2,1,1]
                h_contour[:,0] = [1,1,h1-2,h1-2,1]

        h_contour = np.squeeze(h_contour)
        h_contour = np.asarray(h_contour)

        n_points = h_contour.shape[0]

        w = bbox[0][2]
        h = bbox[0][3]

        try:
            x = h_contour[:,1]
            y = h_contour[:,0]
            okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
            x = np.r_[x[okay], x[-1], x[0]]
            y = np.r_[y[okay], y[-1], y[0]]
        except:
            h_contour = np.zeros((5,2))
            h_contour[:,1] = [1,w1-2,w1-2,1,1]
            h_contour[:,0] = [1,1,h1-2,h1-2,1]
            smoothened = uniformsample(h_contour,200)
            return smoothened


        try:
            tck, u = splprep([x,y], k=1, s=0)
            u_new = np.linspace(u.min(), u.max(), int(200))
            smoothened = np.zeros((int(200),2), dtype = np.float32)
            [smoothened[:,1], smoothened[:,0]] = splev(u_new, tck, der=0)
        except:
            smoothened = uniformsample(h_contour,200)

        original_hull = smoothened


        return original_hull
    except:
        h_contour = np.zeros((5,2))
        h_contour[:,1] = [1,w1-2,w1-2,1,1]
        h_contour[:,0] = [1,1,h1-2,h1-2,1]
        return h_contour
