import torch
import json
import os
import argparse
import numpy as np
import math
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
from tensorboardX import SummaryWriter
from skimage.transform import pyramid_expand
from skimage.io import imsave
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from mrcnn_st import utils
from mrcnn_st import visualize
from mrcnn_st.visualize import display_images
import imageio
import cv2
import gzip
import sys, os
import statistics
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,generate_binary_structure
sys.path.append('/home/abhitrivedi/polyrnn-pp-pytorch/code/')

from functools import partial
import pickle

from Utils import utils
from DataProvider import cityscapes
from Models.Poly import polyrnnpp
from Evaluation import losses, metrics
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # print(result_border+0)
    # result_border = (result_border+0).astype(np.float32)
    # imsave("./test_comp1/" +"truth.jpg", result_border)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    # print(reference_border)
    # reference_border = (reference_border + 0).astype(np.float32)
    # imsave("./test_comp1/" +"truth.jpg", reference_border)
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    # print(dt)
    reference_border = (reference_border + 0).astype(np.float32)
    # imsave("./test_comp1/" +"truth.jpg", reference_border)
    sds = dt[result_border]
    
    return sds

def asd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average surface distance metric.
    
    Computes the average surface distance (ASD) between the binary objects in two images.
    
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
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.
    
    Returns
    -------
    asd : float
        The average surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing
        of elements along each dimension, which is usually given in mm.
        
    See also
    --------
    :func:`assd`
    :func:`hd`
    
    
    Notes
    -----
    This is not a real metric, as it is directed. See `assd` for a real metric of this.
    
    The method is implemented making use of distance images and simple binary morphology
    to achieve high computational speed.
    
    Examples
    --------
    The `connectivity` determines what pixels/voxels are considered the surface of a
    binary object. Take the following binary image showing a cross
    
    >>> from scipy.ndimage.morphology import generate_binary_structure
    >>> cross = generate_binary_structure(2, 1)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])
           
    With `connectivity` set to `1` a 4-neighbourhood is considered when determining the
    object surface, resulting in the surface
    
    .. code-block:: python
    
        array([[0, 1, 0],
               [1, 0, 1],
               [0, 1, 0]])
           
    Changing `connectivity` to `2`, a 8-neighbourhood is considered and we get:
    
    .. code-block:: python
    
        array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]])
           
    , as a diagonal connection does no longer qualifies as valid object surface.
    
    This influences the  results `asd` returns. Imagine we want to compute the surface
    distance of our cross to a cube-like object:
    
    >>> cube = generate_binary_structure(2, 1)
    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]])
           
    , which surface is, independent of the `connectivity` value set, always
    
    .. code-block:: python
    
        array([[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]])
           
    Using a `connectivity` of `1` we get
    
    >>> asd(cross, cube, connectivity=1)
    0.0
    
    while a value of `2` returns us
    
    >>> asd(cross, cube, connectivity=2)
    0.20000000000000001
    
    due to the center of the cross being considered surface as well.
    
    """
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd

def assd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average symmetric surface distance.
    
    Computes the average symmetric surface distance (ASD) between the binary objects in
    two images.
    
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
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.         
        
    Returns
    -------
    assd : float
        The average symmetric surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.
        
    See also
    --------
    :func:`asd`
    :func:`hd`
    
    Notes
    -----
    This is a real metric, obtained by calling and averaging
    
    >>> asd(result, reference)
    
    and
    
    >>> asd(reference, result)
    
    The binary images can therefore be supplied in any order.
    """
    assd = np.mean( (asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)) )
    return assd

def hausd(result, reference, voxelspacing=None, connectivity=1):
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

def hausd95(result, reference, voxelspacing=None, connectivity=1):
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

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    return args

def get_data_loaders(opts, DataProvider):
    print('Building dataloaders')

    dataset_train = DataProvider(split='train', opts=opts['train'])
    dataset_val = DataProvider(split='train_val', opts=opts['train_val'])

    weights, label_to_count = dataset_train.getweight()
    print(len(weights))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(dataset_train, batch_size=opts['train']['batch_size'],sampler =sampler,
        shuffle = False, num_workers=opts['train']['num_workers'], collate_fn=cityscapes.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=opts['train_val']['batch_size'],
        shuffle = False, num_workers=opts['train_val']['num_workers'], collate_fn=cityscapes.collate_fn)
    
    return train_loader, val_loader

class Trainer(object):
    def __init__(self, args):
        self.global_step = 0
        self.epoch = 0
        self.opts = json.load(open(args.exp, 'r'))
        utils.create_folder(os.path.join(self.opts['exp_dir'], 'checkpoints'))

        # Copy experiment file
        os.system('cp %s %s'%(args.exp, self.opts['exp_dir']))

        self.writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train_val'))

        self.train_loader, self.val_loader = get_data_loaders(self.opts['dataset'], cityscapes.DataProvider)
        self.model = polyrnnpp.PolyRNNpp(self.opts).to(device)


        self.grid_size = self.model.encoder.feat_size

        if 'encoder_reload' in self.opts.keys():
            self.model.encoder.reload(self.opts['encoder_reload'])

        # OPTIMIZER
        no_wd = []
        wd = []
        print('Weight Decay applied to: ')

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                # No optimization for frozen params
                continue

            if 'bn' in name or 'conv_lstm' in name or 'bias' in name:
                no_wd.append(p)
            else:
                wd.append(p)
                print(name)

        # Allow individual options
        self.optimizer = optim.Adam(
            [
                {'params': no_wd, 'weight_decay': 0.0},
                {'params': wd}
            ],
            lr=self.opts['lr'], 
            weight_decay=self.opts['weight_decay'],
            amsgrad=False)
        # TODO: Test how amsgrad works (On the convergence of Adam and Beyond)

        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts['lr_decay'], 
            gamma=0.1)

        if args.resume is not None:
            self.resume(args.resume)
            
    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints', 'epoch%d_step%d.pth'\
        %(epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model')

    def resume(self, path):
        self.model.reload(path)
        # self.model.load_state_dict(torch.load('/home/abhitrivedi/polyrnn-pp-pytorch/models/mle_epoch9_step49000.pth')["state_dict"])
        save_state = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])

        print('Model reloaded to resume from Epoch %d, Global Step %d from model at %s'%(self.epoch, self.global_step, path)) 

    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            self.epoch = epoch
            # self.save_checkpoint(epoch) 
            self.lr_decay.step()
            print('LR is now: ', self.optimizer.param_groups[0]['lr'])
            self.train(epoch)

    def train(self, epoch):
        print('Starting training')
        self.model.train()

        accum = defaultdict(float)
        # To accumulate stats for printing

        for step, data in enumerate(self.train_loader):
            if self.global_step % self.opts['val_freq'] == 0 and self.global_step>0:
                self.validate(4)
                if epoch %2 == 0:
                    self.save_checkpoint(epoch)             

            # Forward pass
            output = self.model(data['img'].to(device), data['fwd_poly'].to(device))
            
            # Smoothed targets
            dt_targets = utils.dt_targets_from_class(output['poly_class'].cpu().numpy(),
                self.grid_size, self.opts['dt_threshold'])

            # Get losses
            loss = losses.poly_vertex_loss_mle(torch.from_numpy(dt_targets).to(device), 
                data['mask'].to(device), output['logits'])
            fp_edge_loss = self.opts['fp_weight'] * losses.fp_edge_loss(data['edge_mask'].to(device), 
                output['edge_logits'])
            fp_vertex_loss = self.opts['fp_weight'] * losses.fp_vertex_loss(data['vertex_mask'].to(device), 
                output['vertex_logits'])

            total_loss = loss + fp_edge_loss + fp_vertex_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            if 'grad_clip' in self.opts.keys():
                nn.utils.clip_grad_norm_(self.model.parameters(), self.opts['grad_clip']) 

            self.optimizer.step()

            # Get accuracy
            accuracy = metrics.train_accuracy(output['poly_class'].cpu().numpy(), data['mask'].cpu().numpy(), 
                output['pred_polys'].cpu().numpy(), self.grid_size)

            # Get IoU
            iou = 0
            pred_polys = output['pred_polys'].cpu().numpy()
            gt_polys = data['full_poly']

            for i in range(pred_polys.shape[0]):
                p = pred_polys[i]
                p = utils.get_masked_poly(p, self.grid_size)
                p = utils.class_to_xy(p, self.grid_size)
                i, masks = metrics.iou_from_poly(p, gt_polys[i], self.grid_size, self.grid_size)
                iou += i

            iou = iou / pred_polys.shape[0]

            accum['loss'] += float(loss)
            accum['fp_edge_loss'] += float(fp_edge_loss)
            accum['fp_vertex_loss'] += float(fp_vertex_loss)
            accum['accuracy'] += accuracy
            accum['iou'] += iou
            accum['length'] += 1

            if step % self.opts['print_freq'] == 0:
                # Mean of accumulated values
                for k in accum.keys():
                    if k == 'length':
                        continue
                    accum[k] /= accum['length']

                # Add summaries
                masks = np.expand_dims(masks, -1).astype(np.uint8) # Add a channel dimension
                masks = np.tile(masks, [1, 1, 1, 3]) # Make [2, H, W, 3]
                img = (data['img'].cpu().numpy()[-1,...]*255).astype(np.uint8)
                img = np.transpose(img, [1,2,0]) # Make [H, W, 3]
                vert_logits = np.reshape(output['vertex_logits'][-1, ...].detach().cpu().numpy(), (self.grid_size, self.grid_size, 1))
                edge_logits = np.reshape(output['edge_logits'][-1, ...].detach().cpu().numpy(), (self.grid_size, self.grid_size, 1))
                vert_logits = (1/(1 + np.exp(-vert_logits))*255).astype(np.uint8)
                edge_logits = (1/(1 + np.exp(-edge_logits))*255).astype(np.uint8)
                vert_logits = np.tile(vert_logits, [1, 1, 3]) # Make [H, W, 3]
                edge_logits = np.tile(edge_logits, [1, 1, 3]) # Make [H, W, 3]
                vertex_mask = np.tile(np.expand_dims(data['vertex_mask'][-1,...].cpu().numpy().astype(np.uint8)*255,-1),(1,1,3))
                edge_mask = np.tile(np.expand_dims(data['edge_mask'][-1,...].cpu().numpy().astype(np.uint8)*255,-1),(1,1,3))


                if self.opts['return_attention'] is True:
                    att = output['attention'][-1, 1:4, ...].detach().cpu().numpy()
                    att = np.transpose(att, [0, 2, 3, 1]) # Make [T, H, W, 1]
                    att = np.tile(att, [1, 1, 1, 3]) # Make [T, H, W, 3]
                    def _scale(att):
                        att = att/np.max(att)
                        return (att*255).astype(np.int32)
                
                for k in accum.keys():
                    if k == 'length':
                        continue
                    self.writer.add_scalar(k, accum[k], self.global_step)

                print("[%s] Epoch: %d, Step: %d, Polygon Loss: %f, Edge Loss: %f, Vertex Loss: %f, Accuracy: %f, IOU: %f"\
                %(str(datetime.now()), epoch, self.global_step, accum['loss'], accum['fp_edge_loss'], accum['fp_vertex_loss'],\
                accum['accuracy'], accum['iou']))
                
                accum = defaultdict(float)

            del(output)
            self.global_step += 1

    def validate(self, kt):
        print('Validating')
        self.model.encoder.eval()
        self.model.first_v.eval()
        # Leave LSTM in train mode

        classes = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']
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
            final_ious[clss] = 0.0
            final_acc[clss] = 0.0
            final_hd[clss] = []
            final_hd95[clss] = []

        ious = []
        hd_list = []
        accuracies = []

        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):
                output = self.model(data['img'].to(device), data['fwd_poly'].to(device))

                # Get accuracy
                accuracy = metrics.train_accuracy(output['poly_class'].cpu().numpy(), data['mask'].cpu().numpy(), 
                    output['pred_polys'].cpu().numpy(), self.grid_size)

                # Get IoU
                iou = 0
                hd_f = 0
                pred_polys = output['pred_polys'].cpu().numpy()
                gt_polys = data['full_poly']

                height1 = data['heig'][0]
                # print(height1)
                weight1 = data['weig'][0]
                # print(weight1)

                for i in range(pred_polys.shape[0]):
                    i1 = i
                    p = pred_polys[i]
                    p = utils.get_masked_poly(p, self.grid_size)
                    p = utils.class_to_xy(p, self.grid_size)

                pred = p


                mask = np.zeros((height1, weight1))
                # cv2.fillPoly(mask, np.int32([pred]), [1])
                mask = utils.get_poly_mask(pred, mask)
                # mask = utils

                # print(data['original_mask'][0])
                original_mask = np.asarray(data["original_mask"].cpu().numpy()[0])
                
                original_mask = original_mask.astype(np.uint8)
                original_mask = (original_mask*255).astype(np.uint8)

                pred_mask = mask.astype(np.uint8)
                pred_mask = (pred_mask*255).astype(np.uint8)

                f1 = 0

                hd25 = hausd(pred_mask, original_mask)
                hd925 = hausd95(pred_mask, original_mask)

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
                
                class_lab = data['label'][0]
                
                # final_ious[class_lab] += iou1
                # print(f1)
                final_acc[class_lab] += hd25
                testarr.append(class_lab)

                # palm_leaf_pred1 = copy.deepcopy(data['img_orig'].cpu().numpy()[0])
                palm_leaf_pred1 = data['img_orig'][0]


                hd_list.append(hd25)

                iou = iou / pred_polys.shape[0]
                # hd_list.append(hd_f)
                ious.append(iou)
                accuracies.append(accuracy1)

                del(output)

            iou = np.mean(ious)
            hd = np.mean(hd_list)
            hd_sd = statistics.stdev(hd_list)
            accuracy = np.mean(accuracies)

            self.val_writer.add_scalar('iou', float(iou), self.global_step)
            self.val_writer.add_scalar('HD', float(hd), self.global_step)
            self.val_writer.add_scalar('accuracy', float(accuracy), self.global_step)

            print('[VAL] IoU: %f, Accuracy: %f, Hausdorff: %f, Hausdorff_sd: %f'%(iou, accuracy, hd, hd_sd))

        # Reset
        for ij in testarr:
            testcount[ij] = testcount.get(ij, 0) + 1

        for key in final_acc:
            if int(testcount[key])==0:
                final_acc[key] = 0.0
            else:    
                final_acc[key] /=  testcount[key]

        print("Class-wise Accs: ",final_acc)
        print("Class-wise Accs average: ",np.mean(np.array(list(final_acc.values())).astype(np.float)))
        print('--------------------------------------')

        # self.model.train()

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.loop()
