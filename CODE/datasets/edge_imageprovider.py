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
import imageio
import PIL
import warnings
warnings.filterwarnings("ignore")

import Utils.utils as utils
EPS = 1e-7 

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
      
        if indices is None:
            self.indices = list(range(len(dataset))) 
        else:
            self.indices = indices
            
        if num_samples is None:
            self.num_samples = len(self.indices) 
        else:
            self.num_samples = num_samples
            
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

        # ----------- have a look at the Json format -----------

        df = instance['image_url']
        df1 = df.replace("%20", " ")
        n = 3
        start = df1.find("/")
        while start >= 0 and n > 1:
            start = df1.find("/", start + len("/"))
            n -= 1

        df2 = df1[start:]

        # -----------  path to images -----------

        i_url = "data"+df2

        img = cv2.imread(i_url)
        w_image, h_image = img.shape[1], img.shape[0]

        img1 = imread(i_url)
        
        label = instance['label']
        bbox = component['bbox']
        x0 = max(int(bbox[0]),0)
        y0 = max(int(bbox[1]),0)
        w = max(int(bbox[2]),0)
        h = max(int(bbox[3]),0)
        poly = component["poly"]
        poly2 = component["poly"]


        poly = np.asarray(poly)

        # -----------  expanding the over fitting bbox  -----------

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

        pad_h = 0
        pad_w = 0


        # -----------  padding on image to make dimensions even  -----------
        
        if img.shape[0]%2 != 0:
            pad_h = 1
        if img.shape[1]%2 != 0:
            pad_w = 1


        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)



        bbox1 = bbox

        poly = np.array(poly).astype(np.float)
        poly2 = np.array(poly2).astype(np.float)

        old_poly = copy.deepcopy(poly)

        poly[:,0] = poly[:,0] - x0
        poly[:,1] = poly[:,1] - y0

        if x0 - 6 >= 0 and y0-2 >=0:
            x0 = x0 - 6
            y0 = y0 - 2

        actual_gt_poly11 = poly.tolist()
        actual_gt_poly11 = np.array(actual_gt_poly11)

        poly4 = poly

        poly[:,0] = poly[:,0]/float(w)
        poly[:,1] = poly[:,1]/float(h)

        actual_gt_poly = poly
        actual_gt_poly = np.array(actual_gt_poly)

        eps = 1e-6


        original_mask = np.zeros((h, w),np.float32)
        original_mask = utils.get_original_mask(poly,original_mask)

        poly_length = len(poly)

        img_orig = img1[:, :]

        bh = img.shape[0]
        bw = img.shape[1]
        img = resize(img, (bh, bw,3))

        img = torch.from_numpy(img)

        # -----------  setting up classifier requirements  ----------- 
        gt_label44 = np.zeros((8),np.float32)
        classes = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']
        for i in range(8):
            if label == classes[i]:
                gt_label = i
                gt_label44[i] = 1.0


        return_dict = {
                    "actual_gt_poly": actual_gt_poly,
                    "actual_gt_poly11": actual_gt_poly11,
                    "image_url": i_url,
                    "gt_label": gt_label,
                    "cm_label":label,
                    "file_name": instance['image_url'],
                    "label": instance['label'],
                    "gt_LGCN": gt_label44,
                    "bbox": bbox1,
                    "img":img,
                    "img_orig":img_orig,
                    "poly" : old_poly,
                    "original_mask": original_mask,
                    "x0" : x0,
                    "y0" : y0,
                    "w" : w,
                    "h" : h,
                    "w_image": w_image,
                    "h_image": h_image
                    }
        
        return return_dict
