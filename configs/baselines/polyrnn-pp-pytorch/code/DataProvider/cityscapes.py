import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os.path as osp
import json
import skimage.transform as transform
from skimage import img_as_ubyte
import multiprocessing.dummy as multiprocessing
import random
from skimage.transform import rescale, resize
import Utils.utils as utils
from skimage.io import imsave, imread

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

    examples = []
    skipped_instances = 0

    for instance in ann:
        components = instance['components']

        if 'class_filter'in opts.keys() and instance['label'] not in opts['class_filter']:
            continue
        
        candidates = [c for c in components if len(c['poly']) >= opts['min_poly_len']]
        # candidates = [c for c in components if len(c['poly']) <= opts['max_poly_len']]
        # candidates = [c for c in components]

        # if 'sub_th' in opts.keys():
        #     total_area = np.sum([c['area'] for c in candidates])
        #     candidates = [c for c in candidates if c['area'] > opts['sub_th']*total_area]

        # candidates = [c for c in candidates if c['area'] >= opts['min_area']]

        if opts['skip_multicomponent'] and len(candidates) > 1:
            skipped_instances += 1
            continue

        instance['components'] = candidates
        if candidates:
            examples.append(instance)

    return examples, skipped_instances   

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
    def __init__(self, opts, split='train', mode='train_ce'):
        """
        split: 'train', 'train_val' or 'val'
        opts: options from the json file for the dataset
        """
        self.opts = opts
        self.mode = mode
        self.label_to_count = {}
        self.weight = []
        print('Dataset Options: ', opts)

        if self.mode != 'tool':
            # in tool mode, we just use these functions
            self.data_dir = osp.join(opts['data_dir'], split)
            print("nclnlcnlnlcklacakc", self.data_dir)
            self.instances = []
            self.read_dataset()
            print('Read %d instances in %s split'%(len(self.instances), split))

    def read_dataset(self):
        data_list = glob.glob(osp.join(self.data_dir, '*.json'))
        data_list = [[d, self.opts] for d in data_list]

        pool = multiprocessing.Pool(self.opts['num_workers'])
        data = pool.map(process_info, data_list)
        pool.close()
        pool.join()


        print("Dropped %d multi-component instances"%(np.sum([s for _,s in data])))

        self.instances = [instance for image,_ in data for instance in image]

        self.instances = self.instances[:]
        for instance in self.instances:
            label = instance['label']
            if label not in self.label_to_count:
                self.label_to_count[label]=1
            else:
                # if label == 'Character Line Segment':
                #     self.label_to_count[label]+=1
                # else:
                self.label_to_count[label]+=1
        #list of weights that are inverse of frequency of that class
        self.weight = [1.0/self.label_to_count[instance['label']] for instance in self.instances] 

        # if 'debug' in self.opts.keys() and self.opts['debug']:
        #     self.instances = self.instances[:16]

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

        if self.opts['skip_multicomponent']:
            # Skip_multicomponent is true even during test because we use only
            # 1 bbox and no polys
            assert len(instance['components']) == 1, 'Found multicomponent instance\
            with skip_multicomponent set to True!'

            component = instance['components'][0]
            results = self.prepare_component(instance, component)

        else:
            if self.mode == 'test':
                component = instance['components'][0]
                results = self.prepare_component(instance, component)

            elif 'train' in self.mode:
                component = random.choice(instance['components'])
                results = self.prepare_component(instance, component)

            results['instance'] = instance
            # When we have multicomponents turned on, also send the whole instance
            # In test, this is used to calculate IoU. In train(RL/Evaluator), 
            # this is used to calculate the reward

        return results

    def prepare_component(self, instance, component):
        """
        Prepare a single component within an instance
        """
        get_poly = 'train' in self.mode or 'tool' in self.mode

        if self.opts['flip']:
            example_flip = random.random() >= 0.5
        else:
            example_flip = False

        if self.opts['random_start']:
            random_start = random.random() >= 0.5
        else:
            random_start = False

        lo,hi = self.opts['random_context']
        context_expansion = random.uniform(lo, hi)

        crop_info = self.extract_crop(component, instance, context_expansion)

        img = crop_info['img']

        if example_flip:
            img = np.fliplr(img)
        
        train_dict = {}
        if get_poly:
            poly = crop_info['poly']
            aise_poly = crop_info['aise_poly']

            if example_flip:
                # Flip polygon
                poly[:,0] = 1. - poly[:,0]

            if random_start:
                poly = np.roll(poly, random.choice(range(poly.shape[0])), axis=0)

            orig_poly = poly.copy()

            # Convert from [0, 1] to [0, grid_side]
            poly = utils.poly01_to_poly0g(poly, self.opts['grid_side'])
            fwd_poly = poly

            if 'train' in self.mode:
                # Get masks
                vertex_mask = np.zeros((self.opts['grid_side'], self.opts['grid_side']), np.float32)
                edge_mask = np.zeros((self.opts['grid_side'], self.opts['grid_side']), np.float32)
                # original_mask = np.zeros((crop_info['height'],crop_info['weight'] ), np.float32)

                vertex_mask = utils.get_vertices_mask(poly, vertex_mask)
                edge_mask = utils.get_edge_mask(poly, edge_mask)
                # original_mask = utils.get_poly_mask(poly, original_mask)

                # Don't append first_v to end if in tool mode
                fwd_poly = np.append(fwd_poly, [fwd_poly[0]], axis=0)

            bwd_poly = fwd_poly[::-1]

            if example_flip:
                fwd_poly, bwd_poly = bwd_poly, fwd_poly

            arr_fwd_poly = np.ones((self.opts['max_poly_len'], 2), np.float32) * -1
            arr_bwd_poly = np.ones((self.opts['max_poly_len'], 2), np.float32) * -1
            arr_mask = np.zeros(self.opts['max_poly_len'], np.int32)

            len_to_keep = min(len(fwd_poly), self.opts['max_poly_len'])

            arr_fwd_poly[:len_to_keep] = fwd_poly[:len_to_keep]
            arr_bwd_poly[:len_to_keep] = bwd_poly[:len_to_keep]
            arr_mask[:len_to_keep+1] = 1
            # Numpy doesn't throw an error if the last index is greater than size

            train_dict = {
                'fwd_poly': arr_fwd_poly,
                'bwd_poly': arr_bwd_poly,
                'mask': arr_mask,
                'orig_poly': orig_poly,
                'full_poly': fwd_poly,
            }

            if 'train' in self.mode:
                train_dict['vertex_mask'] = vertex_mask
                train_dict['edge_mask'] = edge_mask
                # train_dict['original_mask'] = original_mask
                train_dict['label'] = instance['label']

        # for Torch, use CHW, instead of HWC
        img = img.transpose(2,0,1)

        original_mask = np.zeros((crop_info['height'],crop_info['weight'] ), np.float32)
        original_mask = utils.get_poly_mask77(aise_poly, original_mask)
        # print(original_mask)

        return_dict = {
            'img': img,
            'img_orig':crop_info['img_orig'],
            'img_path': crop_info['rtrt'],
            'patch_w': crop_info['patch_w'],
            'starting_point': crop_info['starting_point'],
            'heig' : crop_info['height'],
            'weig' : crop_info['weight'],
            'original_mask': original_mask
        }

        return_dict.update(train_dict)

        return return_dict

    def extract_crop(self, component, instance, context_expansion):
        
        df = instance['image_url']
        label1 = instance['label']
        df1 = df.replace("%20", " ")
        n = 3
        start = df1.find("/")
        while start >= 0 and n > 1:
            start = df1.find("/", start + len("/"))
            n -= 1

        df2 = df1[start:]
        i_url = "/doc_images/new_jpg_data"+df2

        img = utils.rgb_img_read(i_url)
        img1 = imread(i_url)

        # print(img)

        get_poly = 'train' in self.mode or 'tool' in self.mode

        if get_poly:
            poly = np.array(component['poly'])
            poly1 = np.array(component['poly'])
            # print(poly.shape[0])
            # if poly.shape[0] > 97:
            poly = uniformsample(poly, 97)

        #     xs = poly[:,0]
        #     ys = poly[:,1]

        bbox = component['bbox']

        x0 = max(int(bbox[0]),0)
        y0 = max(int(bbox[1]),0)
        w = max(int(bbox[2]),0)
        h = max(int(bbox[3]),0)


        # x0, y0, w, h = bbox
        if x0 - 6 >= 0 and y0-2 >=0:
            # x0 = x0 - 6
            # y0 = y0 - 2
            img = img[y0-2:y0+h+2,x0-6:x0+w+6]
            img1 = img1[y0-2:y0+h+2,x0-6:x0+w+6]
            w = w+12
            h = h+4
            poly[:,0] = poly[:,0] + 6
            poly[:,1] = poly[:,1] + 2
            poly1[:,0] = poly1[:,0] + 6
            poly1[:,1] = poly1[:,1] + 2
        else:
            img = img[y0:y0+h,x0:x0+w]
            img1 = img1[y0:y0+h,x0:x0+w]

        # if get_poly:
        # poly = np.array(component['poly'])

        poly1[:,0] = poly1[:,0] - x0
        poly1[:,1] = poly1[:,1] - y0

        xs = poly[:,0]
        ys = poly[:,1]

        x_center = x0 + (1+w)/2.
        y_center = y0 + (1+h)/2.

        widescreen = True if w > h else False
        
        # if not widescreen:
        #     img = img.transpose((1, 0, 2))
        #     x_center, y_center, w, h = y_center, x_center, h, w
        #     if get_poly:        
        #         xs, ys = ys, xs

        # x_min = int(np.floor(x_center - w*(1 + context_expansion)/2.))
        # x_max = int(np.ceil(x_center + w*(1 + context_expansion)/2.))
        
        # x_min = max(0, x_min)
        # x_max = min(img.shape[1] - 1, x_max)

        patch_w = w
        patch_h = h
        # NOTE: Different from before

        # y_min = int(np.floor(y_center - patch_w / 2.))
        # y_max = y_min + patch_w

        top_margin = 0

        # y_min = max(0, y_min)
        # y_max = min(img.shape[0] - 1, y_max)

        scale_factor = float(self.opts['img_side'])/patch_w

        # if x0 - 6 >= 0 and y0-2 >=0:
        #     img = img[y0-2:y0+h+2,x0-6:x0+w+6]
        #     img1 = img1[y0-2:y0+h+2,x0-6:x0+w+6]
        # else:
        patch_img = img
        img_orig = img1
        # imsave("./test_comp1/" + "ab.jpg", patch_img)
        # new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
        # new_img[top_margin: top_margin + patch_img.shape[0], :, ] = patch_img

        # new_img = transform.rescale(new_img, scale_factor, order=1, 
        #     preserve_range=True, multichannel=True)
        # print(h,w)
        # img_orig = patch_img
        new_img = resize(patch_img, (224, 224,3))
        new_img = new_img.astype(np.float32)
        #assert new_img.shape == [self.opts['img_side'], self.opts['img_side'], 3]

        starting_point = [x0, y0]

        if get_poly:
            xs = (xs - x0) / float(patch_w)
            ys = (ys - y0) / float(patch_h)
            xs = np.clip(xs, 0 + EPS, 1 - EPS)
            ys = np.clip(ys, 0 + EPS, 1 - EPS)
            # print(xs,ys)
        # imsave("./test_comp1/" + ".jpg", patch_img)

            # xs = np.clip(xs, 0 + EPS, 1 - EPS)
            # ys = np.clip(ys, 0 + EPS, 1 - EPS)

        # if not widescreen:
        #     # Now that everything is in a square
        #     # bring things back to original mode
        #     new_img = new_img.transpose((1,0,2))
        #     starting_point = [y_min-top_margin, x_min]
        #     if get_poly:
        #         xs, ys = ys, xs

        return_dict = {
            'img': new_img,
            'img_orig': img_orig,
            'patch_w': patch_w,
            'top_margin': top_margin,
            'patch_shape': patch_img.shape,
            'scale_factor': scale_factor,
            'starting_point': starting_point,
            'widescreen': widescreen,
            'rtrt': i_url,
            'height':h,
            'weight':w,
            'label': label1
        }

        if get_poly:
            poly = np.array([xs, ys]).T
            return_dict['poly'] = poly
        return_dict['aise_poly'] = poly1

        return return_dict
