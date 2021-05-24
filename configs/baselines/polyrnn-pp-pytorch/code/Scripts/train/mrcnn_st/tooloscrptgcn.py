import torch
import torch.nn as nn
from edgeonlyy1 import Model as edge_model
import numpy as np
from GNN import polygnn5
import ConcaveHull as ch
import cv2
from collections import OrderedDict
import json
from test_hull5 import testing_hull


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

def float_to_int(array):
    int_array = array.astype(float, casting='unsafe', copy=True)
    # if not np.equal(array, int_array).all():
    #     raise TypeError("Cannot safely convert float array to int dtype. "
    #                     "Array must only contain whole numbers.")
    return int_array

def angles_in_ellipse(num, a, b):
    angles = np.arange(num)
    angles = angles * 2 * np.pi / num
    angles = angles[::-1]
    return angles


def get_hull(hull, bbox):


    original_hull = []

    binary_hull = []
    feature_hull = []

    w = bbox[2]
    h = bbox[3]

    for i in hull:
        # original_hull.append([i[1]*256 / h, i[0]*960 / w])
        original_hull.append([int(i[1] * h / 30), int(i[0] * w / 60)])
        # print(original_hull)
        binary_hull.append([i[1] / 30, i[0] / 60])
        feature_hull.append([int(i[1]), int(i[0])])
    # print(original_hull)
    return original_hull, binary_hull, feature_hull

def convert_hull_to_cv(hull, bbox):

    original_hull = []

    w = bbox[2]
    h = bbox[3]

    for i in hull:
        original_hull.append([int((i[1]) * w / 30), int((i[0]) * h / 16)])
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



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.edgemodel = edge_model(960, 1920, 3).to(device)
        # self.model_path = './checkpoints_cgcn/Final.pth'
  #       state_dict = torch.load('./checkpoints_cgcn/Final.pth')
  #       new_state_dict = OrderedDict()
  #       for k, v in state_dict["gcn_state_dict"].items():
  #           name = k[7:] # remove `module.`
  #           # print(k,v)
  #           new_state_dict[name] = v
		# # load params
  #       self.edgemodel.load_state_dict(new_state_dict)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     self.edgemodel = nn.DataParallel(self.edgemodel)
        # self.edgemodel.to(device)
        # self.edgemodel.load_state_dict(torch.load(self.model_path)["gcn_state_dict"])
        ct = 0
        for child in self.edgemodel.children():
            ct += 1
            # if ct < 12 or ct >15:
            if ct >0:
                for param in child.parameters():
                    param.requires_grad = False



        """
               state_dim is number of input features to fc layer in GCN,
               (number of channels you feed = fdim )
               """
        state_dim = 129

        self.gcn_model = polygnn5.PolyGNN(state_dim=state_dim,
                                          n_adj= 4 ,
                                          cnn_feature_grids=[1, 129, 30, 60],
                                          coarse_to_fine_steps= 2,
                                          get_point_annotation=False,
                                          ).to(device)
        ct = 0
        for child in self.gcn_model.children():
            ct += 1
            if ct>=1:
                for param in child.parameters():
                    param.requires_grad = False


    def forward(self, img, bbox, dp):
        # temp_hull2 = hull
        # gt_mask2 = gt_mask
        # bbox = bbox.tolist()
        # hull = hull.tolist()
        # gt_mask = gt_mask.tolist()
        # img = torch.cat(img)
        cp = 0

        # state_dict = torch.load('./checkpoints_cgcn/LinesBest.pth')
        # new_state_dict = OrderedDict()
        # for k, v in state_dict["gcn_state_dict"].items():
        #     name = k[7:] # remove `module.`
        #     # print(k,v)
        #     new_state_dict[name] = v
        # # load params
        # self.edgemodel.load_state_dict(new_state_dict)

        tg2, poly_logits, class_prob = self.edgemodel(img.to(device))


        # print("cmcmcmcc",tg2.shape)
        
       
        # batch_ellipse = []
        # for i in range(len(bbox)):
        #     w = bbox[0][2]
        #     h = bbox[0][3]

        #     angles = angles_in_ellipse(100, h, w)
        #     x = w / 2 * np.cos(angles) + w / 2
        #     y = h / 2 * np.sin(angles) + h / 2

        #     x = torch.from_numpy(x)
        #     y = torch.from_numpy(y)

        #     ellipse = torch.stack((x, y))
        #     ellipse = ellipse.permute((1, 0)).numpy()
        #     ellipse = torch.from_numpy(uniformsample(ellipse, 350))

        #     batch_ellipse.append(ellipse)

        # batch_ellipse = torch.stack(batch_ellipse)

        # batch_ellipse = batch_ellipse.tolist()

        # hull_from_data = batch_ellipse
        
        # def compute_iou_and_accuracy(arrs, edge_mask1):
        #     intersection = cv2.bitwise_and(arrs, edge_mask1)
        #     union = cv2.bitwise_or(arrs, edge_mask1)

        #     intersection_sum = np.sum(intersection)
        #     union_sum = np.sum(union)

        #     iou = intersection_sum / union_sum

        #     total = np.sum(arrs)
        #     correct_predictions = intersection_sum

        #     accuracy = correct_predictions / total

        #     return iou, accuracy

        # mask4 = np.zeros((30, 200))
        # pred_mask4 = cv2.fillPoly(mask4, np.int32([hull4]), [1])

        original_hull = []
        binary_hull = []
        feature_hull = []
        listpp = []
        listpp11 = []
        edge_logits = poly_logits
        # edge_logits = edge_logits[:,0,:,:]

        new_list = []


        # new_list22 = arrs1
        for i in range(edge_logits.shape[0]):
            # print(hull)
            # temp_hull = temp_hull2.cpu().numpy()[i]
            # gt_mask4 = gt_mask2.cpu().numpy()[i]
            
            # w = max(int(bbox[i][2]),0)
            # h = max(int(bbox[i][3]),0)
            
            # temp_hull[:,0] = temp_hull[:,0]/w*200
            # temp_hull[:,1] = temp_hull[:,1]/h*30
            
            # mask4 = np.zeros((30, 200))
            # bc1 = temp_hull
            # pred_mask4 = cv2.fillPoly(mask4, np.int32([bc1]), [1])
            
            # # mask5 = np.zeros((30, 200))
            # # bc2 = gt_mask4
            # # gt_mask4 = cv2.fillPoly(mask5, np.int32([bc2]), [1])
            
            # gt_mask4 = gt_mask4.astype(np.uint8)
            # pred_mask4 = pred_mask4.astype(np.uint8)
            
            # iou1, accuracy1 = compute_iou_and_accuracy(pred_mask4, gt_mask4)
            # print(iou1)
            # if accuracy1 > 0.200:
            hull1 = testing_hull(poly_logits,class_prob, bbox[i])
            new_list1 = np.asarray(hull1)

            new_list1 = uniformsample(new_list1,1000)
            # new_list1 = new_list1.tolist()
                
            # new_list1 = np.asarray(new_list1)
            # if clockwise_check(new_list1) == False:
            #     new_list1 = new_list1[::-1]
            new_list.append(new_list1)
        new_list = np.asarray(new_list)


        # if cp == 1:
        #     for i in range(edge_logits.shape[0]):
        #         original_hull_i, binary_hull_i, feature_hull_i = get_hull(new_list[i], bbox[i])
        #         original_hull.append(original_hull_i)
        #         binary_hull.append(binary_hull_i)
        #         feature_hull.append(feature_hull_i)
                # print("ellipse")
        # else:
        for i in range(edge_logits.shape[0]):
            # print(new_list[i])
            original_hull_i, binary_hull_i, feature_hull_i = get_hull(new_list[i], bbox[i])
            original_hull.append(original_hull_i)
            binary_hull.append(binary_hull_i)
            feature_hull.append(feature_hull_i)
                # print("hull")


        ####################################################################################################

        feature_hull = torch.from_numpy(np.asarray(feature_hull))
        original_hull = torch.from_numpy(np.asarray(original_hull))
        binary_hull = torch.from_numpy(np.asarray(binary_hull))
        bbox = torch.from_numpy(np.asarray(bbox))

        output_dict = self.gcn_model(tg2, feature_hull, original_hull,
                                     binary_hull, bbox, dp)

        return output_dict, poly_logits, class_prob
