import torch
import json
import os
import argparse
import copy
import torch.nn as nn
from datetime import datetime
import math
import cv2
import numpy as np
import sklearn.metrics as sm
import sys
import imageio
from Utils import utils
from models.combined_model import Model
import warnings
from skimage.transform import rescale, resize

warnings.filterwarnings("ignore")

cv2.setNumThreads(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("!!!!!!got cuda!!!!!!!")
else:
    print("!!!!!!!!!!!!no cuda!!!!!!!!!!!!")



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)

    args = parser.parse_args()

    return args



class Tester(object):
    def __init__(self, args, opts):
        self.global_step = 0
        self.epoch = 0
        self.opts = opts

        self.model_path = 'checkpoints/Final.pth'

        self.model = Model(self.opts)
        self.model.load_state_dict(torch.load(self.model_path)["gcn_state_dict"])

    def loop(self):
        self.testing()

    def testing(self):
        print('Testing')
        classes = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']

        img_link = self.opts["image_link"]

        doc_img = cv2.imread(img_link)
        w_image, h_image = doc_img.shape[1], doc_img.shape[0]

        vis_img = cv2.imread(img_link)

        bboxs = self.opts["BBox_cordinates"]

        count = 0
        for bbox in bboxs:

            count += 1

            x0 = max(int(bbox[0]),0)
            y0 = max(int(bbox[1]),0)
            w = max(int(bbox[2]),0)
            h = max(int(bbox[3]),0)

            # ---- creating a loose-fit bbox (you can also try only with 'else' part) ---- 
            if x0 - 6 >= 0 and y0-2 >=0:    
                img = doc_img[y0-2:y0+h+2,x0-6:x0+w+6]
                vis_img = doc_img[y0-2:y0+h+2,x0-6:x0+w+6]
                bbox[2] = bbox[2] + 12
                bbox[3] = bbox[3] + 4
                w = w + 12
                h = h + 4
            else:
                img = doc_img[y0:y0+h,x0:x0+w]
                vis_img = doc_img[y0:y0+h,x0:x0+w]


            bh = img.shape[0]
            bw = img.shape[1]
            img = resize(img, (bh, bw,3))

            vis_img1 = copy.deepcopy(vis_img)

            img = torch.from_numpy(img)

            img = img.view(-1, img.shape[0], img.shape[1], 3)
            img = torch.transpose(img, 1, 3)
            img = torch.transpose(img, 2, 3)
            img = img.float()

            bbox = torch.from_numpy(np.asarray([bbox])).to(device)
            
            output_dict, poly_logits, class_prob = self.model(img, bbox, 'val')

            pred_cps = output_dict['pred_polys'][-1]

            n_points = output_dict['n_points']

            pred_cps5 = pred_cps[0]


            pred_x = (pred_cps5[:, 0] * h).view(n_points,1)
            pred_y = (pred_cps5[:, 1] * w).view(n_points,1)

            pred = torch.cat((pred_y, pred_x), dim=1)
            pred = pred.detach().cpu().numpy()
            pred = np.asarray(pred)
            
            mask = np.zeros((h, w))
            cv2.fillPoly(mask, np.int32([pred]), [1])


            pred_mask = mask.astype(np.uint8)
            pred_mask = (pred_mask*255).astype(np.uint8)

            # ----------- Saving Images -----------
            cv2.polylines(vis_img1, np.int32([pred]), True, (210,0,0), thickness=1)
            for point in pred:
                cv2.circle(vis_img1, (int(point[0]), int(point[1])), 1, (0, 0, 210), -1)

            cv2.imwrite("visualization/test_custom_img/" + str(count) + ".jpg", vis_img1)


if __name__ == '__main__':
    args = get_args()
    
    opts = json.load(open(args.exp, 'r'))
    tester = Tester(args, opts)
    tester.loop()
