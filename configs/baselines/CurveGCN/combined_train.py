import torch
import json
import os
import argparse
import copy
import imageio
from active_spline import ActiveSplineTorch
import torch.optim as optim
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,generate_binary_structure
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
import operator
from functools import reduce
from tqdm import tqdm
import edge_imageprovider as image_provider
from losses import poly_mathcing_loss
import cv2
import numpy as np
import torch.nn.functional as F
from combined_model import Model
from Handroff_loss import AveragedHausdorffLoss
from skimage.io import imsave
import matplotlib.pyplot as plt
from torch.autograd import Variable
# from Evaluation.DiffRender.py2d import diff_render_loss
import math
import ConcaveHull as ch
import PIL
import warnings
warnings.filterwarnings("ignore")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("!!!!!!got cuda!!!!!!!")
else:
    print("!!!!!!!!!!!!no cuda!!!!!!!!!!!!")

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.
    
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
        
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.
        
    See also
    --------
    :func:`assd`
    :func:`asd`
    
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd

def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95



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

def sort_clockwise(poly):
    coords = poly[:]
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    coords = sorted(coords, key=lambda coord: (-225 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)

    return coords

def angle(dir):
    """
    Returns the angles between vectors.

    Parameters:
    dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

    The return value is a 1D-array of values of shape (N-1,), with each value
    between 0 and pi.

    0 implies the vectors point in the same direction
    pi/2 implies the vectors are orthogonal
    pi implies the vectors point in opposite directions
    """
    dir2 = dir[1:]
    dir1 = dir[:-1]
    return np.arccos((dir1*dir2).sum(axis=1)/(
        np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))


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



def create_folder(path):
    # if os.path.exists(path):
    # resp = input 'Path %s exists. Continue? [y/n]'%path
    #    if resp == 'n' or resp == 'N':
    #       raise RuntimeError()

    # else:
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

    #done

    train_split = 'train'
    train_val_split ='train_val'

    dataset_train = DataProvider(split=train_split, opts=opts[train_split], mode=train_split)
    dataset_val = DataProvider(split=train_val_split, opts=opts[train_val_split], mode=train_val_split)
    weights, label_to_count = dataset_train.getweight()
    print(len(weights))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(dataset_train, batch_size=opts[train_split]['batch_size'],
                              sampler = sampler, shuffle=False, num_workers=opts[train_split]['num_workers'],
                              collate_fn=image_provider.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=opts[train_val_split]['batch_size'],
                            shuffle=False, num_workers=opts[train_val_split]['num_workers'],
                            collate_fn=image_provider.collate_fn)

    return train_loader, val_loader, label_to_count


class Trainer(object):
    def __init__(self, args, opts):
        self.global_step = 0
        self.epoch = 0
        self.opts = opts

        self.model_path = './checkpoints_cgcn/Curve_GCN2.pth'

        self.handroff_loss = AveragedHausdorffLoss()
        self.spline = ActiveSplineTorch(20, 1300, device=device, alpha=0.5)
        self.model = Model(self.opts)
        #if torch.cuda.device_count() > 1:
           # print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            #device_ids = [i for i in range(torch.cuda.device_count())]
            #self.model = nn.DataParallel(self.model, device_ids)

        # self.model.to(device)


        self.model.to(device)
        self.model.load_state_dict(torch.load(self.model_path)["gcn_state_dict"])



        self.train_loader, self.val_loader, self.label_to_count = get_data_loaders(self.opts['dataset'], image_provider.DataProvider)
        # print(self.label_to_count)
        # c1 = self.label_to_count["Hole(Physical)"]
        # c2 = self.label_to_count["Character Line Segment"]
        # c3 = self.label_to_count["Character Component"]
        # c4 = self.label_to_count["Picture"]
        # c5 = self.label_to_count["Decorator"]
        # c6 = self.label_to_count["Library Marker"]
        # # c7 = self.label_to_count["Physical Degradation"]
        # c8 = c1 + c2 + c3 + c4 + c5 + c6
        # for i in range(len(self.label_to_count)):
        #     c10 = c5/c1
        #     c20 = c5/c2
        #     c30 = c5/c3
        #     c40 = c5/c4
        #     c50 = c5/c5
        #     c60 = c5/c6
        #     # c70 = c5/c7
        #     # c70 = c6/c7
        # self.class_weight = torch.tensor([c10,c20,c30,c40,c50,c60]).to(device)




        self.p_n = torch.ones([28,28], dtype= torch.float32)
        self.p_n = (self.p_n*8).to(device)
        self.edge_loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.p_n)
        # self.edge_loss_fn2 = nn.CrossEntropyLoss()

        self.p_n2 = torch.ones([28,28], dtype= torch.float32)
        self.p_n2 = (self.p_n2*0.7).to(device)
        self.poly_loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.p_n2)

        self.p_n1 = torch.ones([28,28], dtype= torch.float32)
        self.p_n1 = (self.p_n1*12).to(device)
        self.vertex_loss_fn = nn.BCEWithLogitsLoss()

        self.fp_loss_fn = nn.MSELoss()
        self.gcn_loss_sum_train = 0
        self.gcn_loss_sum_val = 0
        self.x1 = []
        self.y1 = []
        self.y2 = []


        # OPTIMIZER
        no_wd = []
        wd = []
        print('Weight Decay applied to: ')



        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                # No optimization for frozen params
                continue

            if 'bn' in name or 'bias' in name:
                no_wd.append(p)
            else:
                wd.append(p)
                print(name,)



        # Allow individual options
        self.optimizer = optim.Adam(
            [
                {'params': no_wd, 'weight_decay': 0.0},
                {'params': wd}
            ],
            lr=self.opts['lr'],
            weight_decay=self.opts['weight_decay'],
            amsgrad=False)

        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts['lr_decay'],
                                                  gamma=0.1)
        # if args.resume is not None:
        #     self.resume(args.resume)

    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'gcn_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints_cgcn', 'epoch%d_step%d.pth' \
                                 % (epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model')

    def resume(self, path):
        # self.conv_lstm.reload(path)
        save_state = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])


        print('Model reloaded to resume from Epoch %d, Global Step %d from model at %s' % (
        self.epoch, self.global_step, path))

    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            self.epoch = epoch
            # if self.epoch % 1 == 0 and self.epoch > 0:
            #     self.save_checkpoint(epoch)
            if self.epoch >= 140:
                break
            if epoch >= 1:
                self.x1.append(epoch)
                self.y1.append(self.gcn_loss_sum_train)
                self.y2.append(self.gcn_loss_sum_val)
                if self.epoch % 5 == 0 and self.epoch > 0:
                    plt.plot(self.x1, self.y1, label = "Training loss")
                    plt.plot(self.x1, self.y2, label = "Validation loss")
                    plt.xlabel('epochs')
                    plt.ylabel('Loss')
                    plt.title('GCN Loss')
                    plt.legend()
                    plt.savefig('foo.png') 

            self.validate(4)

            self.train(epoch)

    def train(self, epoch):
        print('Starting training')
        self.model.train()
        losses = []
        gcn_losses = []
        edge_losses = []
        vertex_losses = []
        accum = defaultdict(float)

        for step, data in enumerate(self.train_loader):
            if self.global_step % self.opts['val_freq'] == 0 and self.global_step>0:
                self.validate(5)
            # if self.global_step % 1300 == 0: 
                self.save_checkpoint(epoch)
            # if step <= self.global_step:
            #     continue

            img = data['img']
            img = torch.cat(img)

            img = img.view(-1, 448, 448, 3)
            img = torch.transpose(img, 1, 3)
            img = torch.transpose(img, 2, 3)
            img = img.float()

            # poly_mask = data["poly_mask"]

            bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device)
            hull = torch.from_numpy(np.asarray(data['hull'])).to(device)
            # hull = []
            gt_mask = torch.from_numpy(np.asarray(data['poly_mask'])).to(device)
            
            w1 = torch.tensor(data["w"]).to(device).float()
            h1 = torch.tensor(data["h"]).to(device).float()
            dp = data['actual_gt_poly']
            # new_list1 = np.asarray(hull[i])
            # dp = simplify_coords(, 1)

            dp = uniformsample_batch(dp, 1300)
            dp = sort_clockwise(dp[0])
            if clockwise_check(dp) == False:
                print("csknaclcc")
            dp = (torch.stack(dp)).float()
            dp_x = dp[:, 1].view(-1, 1300, 1)
            dp_x = dp_x/float(h1[0])
            dp_y = dp[:, 0].view(-1, 1300, 1)
            dp_y = dp_y/float(w1[0])
            dp = torch.cat((dp_x, dp_y), dim=2).to(device)
            # print(dp)
            dp44 = data['actual_gt_poly']
            # new_list1 = np.asarray(hull[i])
            # dp = simplify_coords(, 1)
            dp44 = uniformsample_batch(dp44, 1300)
            dp44 = (torch.stack(dp44)).float()
            dp_x44 = dp44[:, :, 1].view(-1, 1300, 1)
            dp_x44 = dp_x44
            dp_y44 = dp44[:, :, 0].view(-1, 1300, 1)
            dp_y44 = dp_y44
            dp44 = torch.cat((dp_x44, dp_y44), dim=2).to(device)

            orig_sizes = torch.zeros((1,2), dtype = torch.float32)
            orig_sizes[0,0] = float(h1[0])
            orig_sizes[0,1] = float(w1[0])
            # print(dp)
            # dp = torch.tensor(dp).to(device)

            output_dict, vertex_logits, poly_logits = self.model(img.to(device), bbox,hull, gt_mask,dp)


            edge_mask1 = data["edge_mask1"]
            poly_mask = data["poly_mask"]
            vertex_mask = data["vertices_mask"]

            gt_label = torch.tensor(data["gt_label"])

            self.optimizer.zero_grad()




            poly_loss = self.edge_loss_fn(poly_logits[:,0,:,:], edge_mask1.to(device))
            vertex_loss = self.poly_loss_fn(vertex_logits[:,0,:,:], poly_mask.to(device))


            pred_cps = (output_dict['pred_polys'][-1]).float()
            pred_cps88 = pred_cps

            B, N_old, N_new = pred_cps.shape[0], pred_cps.shape[1], pred_cps.shape[1] * 13
            grid = torch.cat([torch.zeros((B, N_new, 1, 1)),
                              torch.linspace(-1, 1, N_new).unsqueeze(0).repeat((B, 1)).unsqueeze(
                                  -1).unsqueeze(-1)], dim=-1).to(device)

            rows = F.grid_sample(pred_cps88.unsqueeze(1)[:, :, :, 0].unsqueeze(-1), grid).squeeze(1)
            cols = F.grid_sample(pred_cps88.unsqueeze(1)[:, :, :, 1].unsqueeze(-1), grid).squeeze(1)
            pred_cps88 = torch.cat([rows,cols], dim=-1)



            gt_right_order, poly_mathcing_loss_sum = poly_mathcing_loss(1300,
                                                                        pred_cps88,
                                                                        dp,
                                                                        loss_type=self.opts['loss_type'])
            
            gt_y = dp[0,:,:]

            loss_v = poly_mathcing_loss_sum + 200*poly_loss + 200*vertex_loss
            loss_sum = han_loss
            self.gcn_loss_sum_train = poly_mathcing_loss_sum
            edge_loss_sum = poly_loss
            vertex_loss_sum = vertex_loss

            # if loss_gcn99 > 0:
            loss_v.backward()

            if 'grad_clip' in self.opts.keys():
                nn.utils.clip_grad_norm_(self.gcn_model.parameters(), self.opts['grad_clip'])

            self.optimizer.step()


            loss = loss_sum

            losses.append(loss_sum)
            gcn_losses.append(self.gcn_loss_sum_train)
            edge_losses.append(edge_loss_sum)
            vertex_losses.append(vertex_loss_sum)



            accum['loss'] += float(loss)
            accum['gcn_loss'] += float(self.gcn_loss_sum_train)
            accum['edge_loss'] += float(edge_loss_sum)
            accum['vertex_loss'] += float(vertex_loss_sum)
            accum['length'] += 1

            if (step % self.opts['print_freq'] == 0):
                # Mean of accumulated values
                for k in accum.keys():
                    if k == 'length':
                        continue
                    accum[k] /= accum['length']

                print("[%s] Epoch: %d, Step: %d, Loss: %f, GCN Loss: %f, Edge Loss: %f,  Vertex Loss: %f" % (
                str(datetime.now()), epoch, self.global_step, accum['loss'], accum['gcn_loss'], accum['edge_loss'], accum['vertex_loss']))
                accum = defaultdict(float)

            self.global_step += 1
        avg_epoch_loss = 0.0


        for i in range(len(losses)):
            avg_epoch_loss += losses[i]

        avg_epoch_loss = avg_epoch_loss / len(losses)
        self.gcn_loss_sum_train = avg_epoch_loss

        print("Average Epoch %d loss is : %f" % (epoch, avg_epoch_loss))

    def validate(self, f_t):
        print('Validating')
        self.model.eval()
        losses = []
        gcn_losses = []
        edge_losses = []
        vertex_losses = []
        avg_acc = 0.0
        avg_iou = 0.0
        classes = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']
        final_ious = {} 
        final_acc = {}
        final_hd = {} 
        final_hd95 = {}
        testcount = {}
        testarr=[]

        for clss in classes: 
            final_ious[clss] = 0.0
            final_acc[clss] = 0.0
            final_hd[clss] = 0.0
            final_hd95[clss] = 0.0

        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):

                img = data['img']
                img = torch.cat(img)

                img = img.view(-1, 448, 448, 3)
                img = torch.transpose(img, 1, 3)
                img = torch.transpose(img, 2, 3)
                img = img.float()



                bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device)
                hull = torch.from_numpy(np.asarray(data['hull'])).to(device)
                # hull = []
                gt_mask = torch.from_numpy(np.asarray(data['poly_mask'])).to(device)
                # hull = []
                # print(hull.shape)
                w1 = torch.tensor(data["w"]).to(device).float()
                h1 = torch.tensor(data["h"]).to(device).float()
                dp = data['actual_gt_poly']
                dp = uniformsample_batch(dp, 100)
                dp = (torch.stack(dp)).to(device)
                dp_x = dp[:, :, 1].view(-1, 100, 1)
                # print("height",dp_x)
                dp_x = dp_x/float(h1[0])
                dp_y = dp[:, :, 0].view(-1, 100, 1)
                # print("width",dp_y)
                dp_y = dp_y/float(w1[0])
                dp = torch.cat((dp_x, dp_y), dim=2)
                dp = torch.tensor(dp).to(device)
                # print(dp[])

                output_dict, vertex_logits, poly_logits = self.model(img.to(device), bbox, hull, gt_mask, dp)


                edge_mask1 = data["edge_mask1"]
                poly_mask = data["poly_mask"]
                vertex_mask = data["vertices_mask"]
                gt_label = torch.tensor(data["gt_label"])

                self.optimizer.zero_grad()


                poly_loss = self.edge_loss_fn(poly_logits[:,0,:,:], edge_mask1.to(device))
                vertex_loss = self.poly_loss_fn(vertex_logits[:,0,:,:], poly_mask.to(device))

                
                pred_cps = output_dict['pred_polys'][-1]


                pred_cps5 = pred_cps[0]
                # print(pred_cps5)
                # pred_cps1 = pred_cps.cpu().numpy()

                pred_x = (pred_cps5[:, 0] * h1[0]).view(100,1)
                pred_y = (pred_cps5[:, 1] * w1[0]).view(100,1)

                pred = torch.cat((pred_y, pred_x), dim=1)
                pred = pred.cpu().numpy()
                pred = np.asarray(pred)

                mask_h = int(h1[0].cpu().numpy())
                mask_w = int(w1[0].cpu().numpy())
                
                mask = np.zeros((mask_h, mask_w))
                cv2.fillPoly(mask, np.int32([pred]), [1])

                original_mask = np.asarray(data["original_mask"].cpu().numpy()[0])

                # imsave("./test_gcn_pred/" + str(step) + "gt_leaf_borders.jpg", original_mask)

                original_mask = original_mask.astype(np.uint8)
                pred_mask = mask.astype(np.uint8)

                original_mask = original_mask.astype(np.uint8)
                original_mask = (original_mask*255).astype(np.uint8)
                contours1, _ = cv2.findContours(original_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                contours_gt = []
                for i in range(len(contours1)):
                    for j in range(len(contours1[i])):
                        contours_gt.append(contours1[i][j][0].tolist())                


                pred_mask = mask.astype(np.uint8)
                pred_mask = (pred_mask*255).astype(np.uint8)
                contours2, _ = cv2.findContours(pred_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                contours_pr = []
                for i in range(len(contours2)):
                    for j in range(len(contours2[i])):
                        contours_pr.append(contours2[i][j][0].tolist())

                def calc_precision_recall(contours_a, contours_b, threshold):
                    x = contours_a
                    y = contours_b

                    xx = np.array(x)
                    hits = []
                    for yrec in y:
                        d = np.square(xx[:,0] - yrec[0]) + np.square(xx[:,1] - yrec[1]) < (threshold * threshold)
                        hits.append(np.any(d == True))
                    top_count = np.sum(hits)       

                    precision_recall = top_count/len(y)
                    return precision_recall, top_count, len(y)

                precision, numerator, denominator = calc_precision_recall(contours_gt, contours_pr, f_t)
                recall, numerator, denominator = calc_precision_recall(contours_pr, contours_gt, f_t)
                f2 = 2*recall*precision/(recall + precision)


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

                iou1, accuracy1 = compute_iou_and_accuracy(pred_mask, original_mask)
                try:
                    hd1 = hd(pred_mask, original_mask)
                    hd951 = hd95(pred_mask, original_mask)
                except:
                    hd1 = 0
                    hd951 = 0
                avg_iou += iou1
                avg_acc += accuracy1


                palm_leaf_pred1 = copy.deepcopy(data['img_orig'][0])

                # imageio.imwrite("./test_comp/" + str(step) + "CurveGCN.jpg", palm_leaf_pred1, quality=100)

                class_lab = data['label'][0]
                
                final_ious[class_lab] += iou1
                final_acc[class_lab] += f2

                final_hd[class_lab] += hd1
                final_hd95[class_lab] += hd951

                testarr.append(class_lab)

                # if iou1 > 0.8:    
                class_lab = data['label'][0]
                

                
                pred_cps1 = pred_cps

                gt_right_order, poly_mathcing_loss_sum = poly_mathcing_loss(100,
                                                                            pred_cps,
                                                                            dp,
                                                                            loss_type=self.opts['loss_type'])
                gt_y = dp[0,:,:]

                loss_sum = poly_mathcing_loss_sum
                self.gcn_loss_sum_val = poly_mathcing_loss_sum
                edge_loss_sum = poly_loss
                vertex_loss_sum = vertex_loss



                loss = loss_sum
                losses.append(loss)
                gcn_losses.append(self.gcn_loss_sum_val)
                edge_losses.append(edge_loss_sum)
                vertex_losses.append(vertex_loss_sum)
                # fp_losses.append(fp_loss.item())

        avg_epoch_loss = 0.0
        avg_gcn_loss = 0.0
        avg_edge_loss = 0.0
        avg_vertex_loss = 0.0


        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
            avg_gcn_loss += gcn_losses[i]
            avg_edge_loss += edge_losses[i]
            avg_vertex_loss += vertex_losses[i]

        for ij in testarr:
          testcount[ij] = testcount.get(ij, 0) + 1

        for key in final_ious:
            if int(testcount[key])==0:
                final_ious[key] = 0.0
            else:    
                final_ious[key] /=  testcount[key]
        for key in final_acc:
            if int(testcount[key])==0:
                final_acc[key] = 0.0
            else:    
                final_acc[key] /=  testcount[key]

        for key in final_hd:
            if int(testcount[key])==0:
                final_hd[key] = 0.0
            else:    
                final_hd[key] /=  testcount[key]

        for key in final_hd95:
            if int(testcount[key])==0:
                final_hd95[key] = 0.0
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


        avg_epoch_loss = avg_epoch_loss / len(losses)
        avg_gcn_loss = avg_gcn_loss / len(losses)
        avg_edge_loss = avg_edge_loss / len(losses)
        avg_vertex_loss = avg_vertex_loss / len(losses)
        self.gcn_loss_sum_val = avg_gcn_loss

        print(avg_gcn_loss)


        print("Average VAL error is : %f, Average VAL gcn error is : %f, Average VAL edge error is : %f, Average VAL vertex error is : %f" % (avg_epoch_loss, avg_gcn_loss, avg_edge_loss, avg_vertex_loss))
        self.model.train()


if __name__ == '__main__':
    args = get_args()
    
    opts = json.load(open(args.exp, 'r'))
    trainer = Trainer(args, opts)
    trainer.loop()
