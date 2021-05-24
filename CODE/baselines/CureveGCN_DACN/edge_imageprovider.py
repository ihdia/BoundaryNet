import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os.path as osp
import cv2
import json
import math
import multiprocessing.dummy as multiprocessing
import copy
import scipy
from skimage.transform import rescale, resize
from skimage.io import imsave, imread
import PIL

import utils
EPS = 1e-7

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

def process_info(args):
    """
    Process a single json file
    """
    fname, opts = args
    
    with open(fname, 'r') as f:
        ann = json.load(f)
        f.close()
    examples = []
    skipped_instances = 0

    for instance in ann:
        components = instance['components']

        if len(components[0]['poly']) < 3:
            continue

        if 'class_filter'in opts.keys() and instance['label'] not in opts['class_filter']:
            continue

        # if instance['image_url'].find('Bhoomi') == -1:
        #     continue

        candidates = [c for c in components]

        instance['components'] = candidates
        if candidates:
            examples.append(instance)

    return examples, skipped_instances

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    #Samples elements randomly from a given list of indices for imbalanced dataset
    def __init__(self, dataset,weight, indices=None, num_samples=None):
      
         # if indices is not provided, 
        # all elements in the dataset will be considered
        if indices is None:
            self.indices = list(range(len(dataset))) 
        else:
            self.indices = indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        if num_samples is None:
            self.num_samples = len(self.indices) 
        else:
            self.num_samples = num_samples
            
        # weight for each sample
       
        self.weights = torch.DoubleTensor(weight)
        
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=False).tolist())
    def __len__(self):
        return self.num_samples

def collate_fn(batch_list):
    keys = batch_list[0].keys()
    collated = {}

    for key in keys:
        val = [item[key] for item in batch_list]

        t = type(batch_list[0][key])
        
        if t is np.ndarray:
            try:
                val = torch.from_numpy(np.stack(val, axis=0))
            except:
                # for items that are not the same shape
                # for eg: orig_poly
                val = [item[key] for item in batch_list]

        collated[key] = val

    return collated

class DataProvider(Dataset):
    """
    Class for the data provider
    """
    def __init__(self, opts, split='train', mode='train'):
        """
        split: 'train', 'train_val' or 'val'
        opts: options from the json file for the dataset
        """
        self.opts = opts
        self.mode = mode
        self.label_to_count = {}
        self.weight = []
        # if split == 'train_val':
        #     self.split = 'val'
        # else:
        self.split = split
        print('Dataset Options: ', opts)
        if self.mode !='tool':
            # in tool mode, we just use these functions
            self.data_dir = osp.join(opts['data_dir'], split)
            self.instances = []
            self.read_dataset()
            print('Read %d instances in %s split'%(len(self.instances), split))
    def read_dataset(self):
        data_list = glob.glob(osp.join(self.data_dir, '*.json'))
        # print(data_list)
        # if self.split == 'train':
        #     data_list = [data_list[1]]
        # print(data_list)
        data_list = [[d, self.opts] for d in data_list]
        # print(len(data_list))
        pool = multiprocessing.Pool(self.opts['num_workers'])
        data = pool.map(process_info, data_list)
        pool.close()
        pool.join()
        print(len(data))

        print("Dropped %d multi-component instances"%(np.sum([s for _,s in data])))
        
        self.instances = [instance for image,_ in data for instance in image]
        # please
        self.instances = self.instances[:]
        for instance in self.instances:
            label = instance['label']
            if label not in self.label_to_count:
                self.label_to_count[label]=1
            else:
                self.label_to_count[label]+=1
        #list of weights that are inverse of frequency of that class
        self.weight = [1.0/self.label_to_count[instance['label']] for instance in self.instances] 
        if 'debug' in self.opts.keys() and self.opts['debug']:
            self.instances = self.instances[:16]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):

        return self.prepare_instance(idx)
    def getweight(self):
        return self.weight, self.label_to_count
    def getlabeltocount(self):
        return self.label_to_count
    def prepare_instance(self, idx):
        """
        Prepare a single instance, can be both multicomponent
        or just a single component
        """
        instance = self.instances[idx]
        component = instance['components'][0]
        
        results = self.prepare_component(instance, component)
        
        return results

    def prepare_component(self, instance, component):

        df = instance['image_url']
        # print(df)
        df1 = df.replace("%20", " ")
        n = 3
        start = df1.find("/")
        while start >= 0 and n > 1:
            start = df1.find("/", start + len("/"))
            n -= 1

        df2 = df1[start:]

        i_url = "./doc_images/new_jpg_data"+df2
        # i_url = "./train5/train5"+df2

        pt = 0
        img = imread(i_url)
        img1 = imread(i_url)
        # print("ski",img.shape)
        # cv2.imshow("a",img)
        label = instance['label']
        bbox = component['bbox']
        x0 = max(int(bbox[0]),0)
        y0 = max(int(bbox[1]),0)
        w = max(int(bbox[2]),0)
        h = max(int(bbox[3]),0)
        poly = component["poly"]
        poly2 = component["poly"]
        # print(img.shape)
        bbox1 = bbox
        # actual_gt_poly = copy.deepcopy(poly)
        # print(len(poly))

        
        hull = []
        hull_gcn = []
        # print(hull_gcn)
        # hull = []
        poly = np.asarray(poly)

        hull = np.array(hull).astype(np.float)
        hull_gcn = np.array(hull_gcn).astype(np.float)
        

        if x0 - 6 >= 0 and y0-2 >=0:    
            img = img[y0-2:y0+h+2,x0-6:x0+w+6]
            img1 = img1[y0-2:y0+h+2,x0-6:x0+w+6]
            poly[:,0] = poly[:,0] + 6
            poly[:,1] = poly[:,1] + 2
            bbox[2] = bbox[2] + 12
            bbox[3] = bbox[3] + 4
            w = w + 12
            h = h + 4
        else:
            img = img[y0:y0+h,x0:x0+w]
            img1 = img1[y0:y0+h,x0:x0+w]

        
        poly = np.array(poly).astype(np.float)
        poly2 = np.array(poly2).astype(np.float)

        
        old_poly = copy.deepcopy(poly)

        # actual_gt_poly = copy.deepcopy(poly)

        poly[:,0] = poly[:,0] - x0
        poly[:,1] = poly[:,1] - y0

        actual_gt_poly = poly
        actual_gt_poly = np.array(actual_gt_poly)

        poly4 = poly

        poly[:,0] = poly[:,0]/float(w)
        poly[:,1] = poly[:,1]/float(h)


        # print("shapeshape", poly)
        # print(poly[:,0], poly[:,1])

        eps = 1e-6

        poly_mask = np.zeros((28, 28),np.float32)
        poly_mask  = utils.get_poly_mask(poly,poly_mask)

        original_mask = np.zeros((h, w),np.float32)
        original_mask = utils.get_original_mask(poly,original_mask)

        # imsave("./test_gcn_pred4/" + "edge_leaf.jpg", poly_mask)

        poly_mask44 = np.zeros((30, 90),np.float32)
        poly_mask44  = utils.get_poly_mask(poly,poly_mask44)



        edge_mask1 = np.zeros((28,28),np.float32)
        edge_mask1  = utils.get_edge_mask(poly,edge_mask1)

        original_mask = np.zeros((h, w),np.float32)
        original_mask = utils.get_original_mask(poly,original_mask)

        n_ones = np.sum(poly_mask)
        n_zeros = 3000 - n_ones

        w_ones = 1500/n_ones
        w_zeros = 1500/n_zeros

        poly77 = uniformsample(poly, 140)

        vertex_mask = np.zeros((28, 28),np.float32)
        vertex_mask = utils.get_vertices_mask(poly,vertex_mask)

        n_ones_v = np.sum(vertex_mask)
        n_zeros_v = 3000 - n_ones        

        w_ones_v = 1500/n_ones_v
        w_zeros_v = 1500/n_zeros_v

        poly_length = len(poly)


        img_orig = img[:, :]
        # img_orig = torch.from_numpy(img_orig)

        img = resize(img, (448, 448,3))


        img = torch.from_numpy(img)
        # gt_label = np.zeros((6),np.float32)
        classes = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']
        for i in range(8):
            if label == classes[i]:
                gt_label = i

        return_dict = {
                    "actual_gt_poly": actual_gt_poly,
                    "actual_gt_poly11": actual_gt_poly,
                    "image_url": instance['image_url'],
                    "gt_label": gt_label,
                    "file_name": instance['image_url'],
                    "label": instance['label'],
                    "bbox": bbox1,
                    "pt":pt,
                    "img":img,
                    "img_orig":img_orig,
                    "poly" : old_poly,
                    "hull": hull,
                    "hullgcn": hull_gcn,
                    "poly_length": poly_length,
                    "edge_mask1":edge_mask1,
                    "vertices_mask":vertex_mask,
                    "original_mask": original_mask,
                    "poly_mask": poly_mask,
                    "poly_mask44":poly_mask44,
                    "n_ones": w_ones,
                    "n_zeros": w_zeros,
                    "n_ones_v": w_ones_v,
                    "n_zeros_v": w_zeros_v,
                    "x0" : x0,
                    "y0" : y0,
                    "w" : w,
                    "h" : h
                    # "w_image": w_image,
                    # "h_image": h_image
                    }
        
        return return_dict
