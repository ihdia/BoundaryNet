import torch
import torch.nn as nn
from .MCNN import Model as enc_model
import numpy as np
from .AGCN import PolyGNN
import math
from scipy.interpolate import splprep, splev
import cv2
from collections import OrderedDict
from Utils.contourization import testing_hull
import skfmm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_hull(hull, bbox, w1, h1, feat_w1, feat_h1):
    original_hull = []

    binary_hull = []
    feature_hull = []

    w = float(bbox[2])
    h = float(bbox[3])

    h1 = float(h1)
    w1 = float(w1)

    for i in hull:
        original_hull.append([int((i[1])), int((i[0]))])
        binary_hull.append([i[1]/h, i[0]/w])
        feature_hull.append([math.floor(i[1] * feat_h1 / h), math.floor(i[0] * feat_w1 / w)])

    return original_hull, binary_hull, feature_hull


class Model(nn.Module):
    def __init__(self, opts):
        super(Model, self).__init__()
        self.opts = opts

        

        self.edgemodel = enc_model(None, None, 3)
        state_dict = torch.load('checkpoints/Final_encoder.pth')
        print(state_dict.items)
        new_state_dict = OrderedDict()

        if self.opts["enc_freeze"]:
            # ----------- Freezing full Encoder -----------
            ct = 0
            for child in self.edgemodel.children():
                ct += 1
                if ct >0:
                    for param in child.parameters():
                        param.requires_grad = False



        """ state_dim is number of input features to GCN from Encoder,
        (number of channels you feed = fdim ) """
        state_dim = 120

        self.gcn_model = PolyGNN(state_dim=state_dim,
                                          n_adj=self.opts['n_adj'],
                                          cnn_feature_grids=self.opts['cnn_feature_grids'],
                                          coarse_to_fine_steps=self.opts['coarse_to_fine_steps'],
                                          ).to(device)

    def forward(self, img, bbox, mode):
                
        bbox = bbox.tolist()
        cp = 0

        tg2, poly_logits, class_prob = self.edgemodel(img.to(device))

        tg2 = tg2.to(device)
        poly_logits = poly_logits.to(device)
        class_prob = class_prob.to(device)

        feat_h1, feat_w1 = tg2[:,0,:,:].shape[1],tg2[:,0,:,:].shape[2]
        
        h1,w1 = poly_logits[:,0,:,:].shape[1],poly_logits[:,0,:,:].shape[2]


        original_hull = []
        binary_hull = []
        feature_hull = []
        listpp = []
        listpp11 = []
        new_list = []

        for i in range(poly_logits.shape[0]):
            hull1 = testing_hull(poly_logits,class_prob, bbox)
            new_list1 = np.asarray(hull1)
            new_list.append(new_list1)
        new_list = np.asarray(new_list)

        for i in range(poly_logits.shape[0]):
            original_hull_i, binary_hull_i, feature_hull_i = get_hull(new_list[i], bbox[i], w1, h1, feat_w1, feat_h1)
            original_hull.append(original_hull_i)
            binary_hull.append(binary_hull_i)
            feature_hull.append(feature_hull_i)

        feature_hull = torch.from_numpy(np.asarray(feature_hull))
        original_hull = torch.from_numpy(np.asarray(original_hull))
        binary_hull = torch.from_numpy(np.asarray(binary_hull))
        bbox = torch.from_numpy(np.asarray(bbox))

        output_dict = self.gcn_model(tg2, feature_hull, original_hull,
                                     binary_hull, bbox, mode)

        return output_dict, poly_logits, class_prob
