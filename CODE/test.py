import torch
import json
import os
import argparse
import copy
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import datasets.edge_imageprovider as image_provider
import math
import cv2
import numpy as np
import sklearn.metrics as sm
import sys
np.set_printoptions(threshold=sys.maxsize)
import torch.nn.functional as F
import imageio
from Utils import utils
from models.combined_model import Model
import warnings


warnings.filterwarnings("ignore")

cv2.setNumThreads(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("!!!!!!got cuda!!!!!!!")
else:
    print("!!!!!!!!!!!!no cuda!!!!!!!!!!!!")


def create_folder(path):
    os.system('mkdir -p %s' % path)
    print('Experiment folder created at: %s' % path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    return args


def get_data_loaders(opts, DataProvider):
    print('Building dataloaders')

    train_val_split ='test'

    dataset_val = DataProvider(split=train_val_split, opts=opts[train_val_split], mode=train_val_split)


    val_loader = DataLoader(dataset_val, batch_size=opts[train_val_split]['batch_size'],
                            shuffle=False, num_workers=opts[train_val_split]['num_workers'],
                            collate_fn=image_provider.collate_fn)

    return val_loader


class Tester(object):
    def __init__(self, args, opts):
        self.global_step = 0
        self.epoch = 0
        self.opts = opts

        self.model_path = 'checkpoints/Final.pth'

        self.model = Model(self.opts)
        self.model.load_state_dict(torch.load(self.model_path)["gcn_state_dict"])

        self.val_loader = get_data_loaders(self.opts['dataset'], image_provider.DataProvider)


    def loop(self):
        self.testing()


    def testing(self):
        print('Testing')
        self.model.eval()
        avg_acc = 0.0
        avg_iou = 0.0
        classes = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']
        pred_cm = []
        gt_cm = []
        final_ious = {} 
        final_acc = {}
        final_hd = {} 
        final_hd95 = {}
        self.d_iou1 = dict()
        self.d_iou_c1 = dict()
        self.d_accuracy1 = dict()
        self.d_accuracy_c1 = dict()
        testcount = {}
        testarr=[]
        for clss in classes: 
            final_ious[clss] = []
            final_acc[clss] = 0.0
            final_hd[clss] = 0.0
            final_hd95[clss] = 0.0
            iou_list = []
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):

                img = data['img']
                img = torch.cat(img)
                img = img.view(-1, img.shape[0], img.shape[1], 3)
                img = torch.transpose(img, 1, 3)
                img = torch.transpose(img, 2, 3)
                img = img.float()


                bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device)

                w1_img = torch.tensor(data["w"]).to(device).float()
                h1_img = torch.tensor(data["h"]).to(device).float()
                
                dp = data['actual_gt_poly11']
                dp_poly = data['actual_gt_poly']

                output_dict, poly_logits, class_prob = self.model(img, bbox, 'val')


                # ----------- Mask extraction for metrics -----------
                poly_logits88 = torch.sigmoid(poly_logits[0,0,:,:]).cpu().numpy()
                yy = poly_logits88 > 0.5
                yy = yy+0
                poly_logits88 = yy.astype(np.float32)

                poly_mask = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                poly_mask  = utils.get_poly_mask(dp_poly.cpu().numpy()[0],poly_mask)

                edge_mask77 = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                edge_mask77  = utils.get_edge_mask(dp_poly.cpu().numpy()[0],edge_mask77)

                n_poly = (np.sum(poly_mask)).astype(np.float32)

                back_mask = 1.0-poly_mask
                n_back = (np.sum(back_mask)).astype(np.float32)

                w1,h1 = poly_logits[:,0,:,:].shape[1],poly_logits[:,0,:,:].shape[2]


                pred_cps = output_dict['pred_polys'][-1]

                pred_cps7 = pred_cps.detach().cpu().numpy()

                n_points = output_dict['n_points']

                dp = utils.uniformsample_batch(dp, n_points)
                dp7 = dp[0].cpu().numpy()
                dp = (torch.stack(dp)).to(device)
                dp_x = dp[:, :, 1].view(-1, n_points, 1)
                dp_x = dp_x/float(h1_img[0])
                dp_y = dp[:, :, 0].view(-1, n_points, 1)
                dp_y = dp_y/float(w1_img[0])
                dp = torch.cat((dp_x, dp_y), dim=2)
                dp = torch.tensor(dp).to(device)

                dp_vis = dp[0]

                dpf = dp.cpu().numpy()

                pred_cps5 = pred_cps[0]


                pred_x = (pred_cps5[:, 0] * h1_img[0]).view(n_points,1)
                pred_y = (pred_cps5[:, 1] * w1_img[0]).view(n_points,1)

                pred = torch.cat((pred_y, pred_x), dim=1)
                pred = pred.cpu().numpy()
                pred = np.asarray(pred)

                mask_h = int(h1_img[0].cpu().numpy())
                mask_w = int(w1_img[0].cpu().numpy())
                
                mask = np.zeros((mask_h, mask_w))
                cv2.fillPoly(mask, np.int32([pred]), [1])

                palm_leaf_pred = copy.deepcopy(data['img_orig'][0])

                palm_leaf_pred1 = copy.deepcopy(data['img_orig'][0])

                original_mask = np.asarray(data["original_mask"][0])


                original_mask = original_mask.astype(np.uint8)
                original_mask = (original_mask*255).astype(np.uint8)

                pred_mask = mask.astype(np.uint8)
                pred_mask = (pred_mask*255).astype(np.uint8)

                iou1, accuracy1 = utils.compute_iou_and_accuracy(pred_mask, original_mask)
                
                # ----------- Hausdorff Distance metrics -----------
                
                hd1 = utils.hd(pred_mask, original_mask)
                hd951 = utils.hd95(pred_mask, original_mask)
                
                # ----------- Saving Images -----------
                cv2.fillPoly(palm_leaf_pred, np.int32([pred]), (210,0,0))
                cv2.addWeighted(palm_leaf_pred, 0.2, palm_leaf_pred1, 1 - 0.3, 0, palm_leaf_pred1)
                cv2.polylines(palm_leaf_pred1, np.int32([dp7]), True, [255,255,255], thickness=1)
                cv2.polylines(palm_leaf_pred1, np.int32([pred]), True, (210,0,0), thickness=1)
                for point in pred:
                    cv2.circle(palm_leaf_pred1, (int(point[0]), int(point[1])), 1, (0, 0, 210), -1)

                imageio.imwrite("visualization/test_gcn_pred/" + str(step) + ".jpg", palm_leaf_pred1, quality=100)

                class_prob = F.softmax(class_prob)
                class_prob = torch.squeeze(class_prob)
                class_label, index = torch.topk(class_prob, 1)
                classes = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']
                
                label46 = data["gt_label"][0]

                avg_iou += iou1
                avg_acc += hd1

                class_lab = data['label'][0]

                gt_cmr = data['cm_label'][0]

                # ----------- Confusion matrix parameters -----------
                pred_cm.append(classes[index[0]])
                gt_cm.append(gt_cmr)
                

                final_acc[class_lab] += accuracy1
                final_ious[class_lab].append(iou1)

                final_hd[class_lab] += hd1
                final_hd95[class_lab] += hd951

                testarr.append(class_lab)

        cm = sm.confusion_matrix(gt_cm, pred_cm, labels = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker','Boundary Line','Physical Degradation'])
        # -----------  Print confusion matrix ----------- 
        # print(cm)


        for ij in testarr:
          testcount[ij] = testcount.get(ij, 0) + 1

        for key in final_ious:
            if int(testcount[key])==0:
                final_ious[key] = 0.0
            else:    
                final_ious[key] =  np.mean(final_ious[key])
        for key in final_acc:
            if int(testcount[key])==0:
                final_acc[key] = 0.0
            else:    
                final_acc[key] /=  testcount[key]

        for key in final_hd:
            if int(testcount[key])==0:
                final_hd[key] = []
            else:    
                final_hd[key] /=  testcount[key]

        for key in final_hd95:
            if int(testcount[key])==0:
                final_hd95[key] = []
            else:    
                final_hd95[key] /=  testcount[key]

        print("Class-wise IOUs: ",final_ious)
        print("Class-wise IOUs average: ",np.mean(np.array(list(final_ious.values())).astype(np.float)))
        print('--------------------------------------')
        print("Class-wise Accs: ",final_acc)
        print("Class-wise Accs average: ",np.mean(np.array(list(final_acc.values())).astype(np.float)))
        print('--------------------------------------')
        print("Class-wise HD: ",final_hd)
        print("Class-wise HD average: ",np.mean(np.array(list(final_hd.values())).astype(np.float)))
        print('--------------------------------------')
        print("Class-wise HD95: ",final_hd95)
        print("Class-wise HD95 average: ",np.mean(np.array(list(final_hd95.values())).astype(np.float)))
        print('--------------------------------------')


if __name__ == '__main__':
    args = get_args()
    
    opts = json.load(open(args.exp, 'r'))
    tester = Tester(args, opts)
    tester.loop()
