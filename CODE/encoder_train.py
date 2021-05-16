import torch
import json
import os
import argparse
import copy
import imageio
import skfmm
from  scipy import ndimage
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
import datasets.edge_imageprovider as image_provider
from tqdm import tqdm
import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from models.MCNN.encoder_model import Model
from skimage.io import imsave
import math
from Utils import utils
from losses.fm_maps import compute_edts_forPenalizedLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("!!!!!!got cuda!!!!!!!")
    print(torch.cuda.device_count())
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

    train_split = 'train'
    train_val_split ='test'

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

        self.model_path = 'checkpoints/Final_encoder.pth'

        self.model = Model(None,None,3)
        self.model.load_state_dict(torch.load(self.model_path)["gcn_state_dict"])
        self.model.to(device)

        self.train_loader, self.val_loader, self.label_to_count = get_data_loaders(self.opts['dataset'], image_provider.DataProvider)

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

        # ----------- Locking Parameters -----------
        # ct = 0
        # for child in self.model.children():
        #     ct += 1            
        #     if ct>=8 and ct<=9:
        #     #     for param in child.parameters():
        #     #         param.requires_grad = False
        #     #         print(child)



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
                                                  gamma=0.5)
        if args.resume is not None:
            self.resume(args.resume)

    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'gcn_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints', 'epoch_encoder_%d_step%d.pth' \
                                 % (epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model')

    def resume(self, path):
        self.model_path = path
        save_state = torch.load(self.model_path)
        self.model.load_state_dict(save_state["gcn_state_dict"])
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])

        print('Model reloaded to resume from Epoch %d, Global Step %d from model at %s' % (
        self.epoch, self.global_step, path))

    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            self.epoch = epoch
            # self.lr_decay.step()
            print('LR is now: ', self.optimizer.param_groups[0]['lr'])
            self.train(epoch)

    def train(self, epoch):
        print('Starting training')
        self.model.train()
        losses = []
        gcn_losses = []
        poly_losses = []
        class_losses = []
        accum = defaultdict(float)
        losses_hn = []
        for step, data in enumerate(self.train_loader):
            if (self.global_step) % self.opts['val_freq'] == 0:
                self.validate()
                self.save_checkpoint(epoch)

            img = data['img']
            img = torch.cat(img)
            img = img.view(-1, img.shape[0], img.shape[1], 3)
            img = torch.transpose(img, 1, 3)
            img = torch.transpose(img, 2, 3)
            img = img.float()

            self.optimizer.zero_grad()
            

            # ----------- GT formation -----------
            bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device)

            w1 = torch.tensor(data["w"]).view(-1,1).to(device).float()
            h1 = torch.tensor(data["h"]).view(-1,1).to(device).float()

            dp_poly = data['actual_gt_poly']
            dp = torch.tensor(data['actual_gt_poly'], dtype = torch.float32).to(device)

            tg2, poly_logits, class_prob = self.model(img.to(device))

            poly_mask = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
            poly_mask  = utils.get_poly_mask(dp.cpu().numpy()[0],poly_mask)

            edge_mask77 = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
            edge_mask77  = utils.get_edge_mask(dp.cpu().numpy()[0],edge_mask77)
            
            back_mask = 1.0-poly_mask

            n_back = (np.sum(back_mask)).astype(np.float32)
            n_poly = (np.sum(poly_mask)).astype(np.float32)
            
            n_edge = (np.sum(edge_mask77)).astype(np.float32)
            dfp = poly_mask
            
            # ----------- Distance Maps Initialization -----------
            DT_mask = compute_edts_forPenalizedLoss(edge_mask77)
            DT_mask = torch.from_numpy(np.asarray(DT_mask)).to(device)
            DT_mask = DT_mask.float()


            w1,h1 = poly_logits[:,0,:,:].shape[1],poly_logits[:,0,:,:].shape[2]

            self.p_n2 = torch.ones([w1,h1], dtype= torch.float32).to(device)
            self.p_n2 = self.p_n2*(n_back/n_poly)

            # ----------- BCE Loss -----------
            self.poly_loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.p_n2, reduction = 'none')
            
            poly_mask = torch.from_numpy(np.asarray(poly_mask)).to(device)
            poly_mask = poly_mask.view(1,poly_mask.shape[0],poly_mask.shape[1]).to(device)
            
            poly_loss1 = self.poly_loss_fn(poly_logits[:,0,:,:], poly_mask.to(device))
            

            # ----------- Focal Loss -----------

            # pt3 = torch.exp(-poly_loss1)
            # poly_loss1 = ((1-pt3)**2) * poly_loss1

            poly_loss1 = poly_loss1*DT_mask


            poly_loss1 = torch.mean(poly_loss1)
            poly_loss = poly_loss1

            gt_label = torch.tensor(data["gt_label"]).to(device)


            # ----------- classifier loss -----------

            # self.edge_loss_fn1 = nn.CrossEntropyLoss()
            # try:
            #     class_loss1 = self.edge_loss_fn1(class_prob.to(device), gt_label.to(device))
            # except Exception as e:
            #     # print(gt_label)
            #     print(e)
            #     continue

            class_loss1 = 0

            loss_v = poly_loss

            loss_sum = poly_loss
            poly_loss_sum = poly_loss
            class_loss_sum = class_loss1

            loss_v.backward()

            if 'grad_clip' in self.opts.keys():
                nn.utils.clip_grad_norm_(self.model.parameters(), self.opts['grad_clip'])

            self.optimizer.step()

            loss = loss_sum

            losses.append(loss)
            poly_losses.append(poly_loss_sum)
            class_losses.append(class_loss_sum)

            accum['loss'] += float(loss)
            accum['poly_loss'] += float(poly_loss_sum)
            accum['class_loss'] += float(class_loss_sum)
            accum['length'] += 1

            if (step % self.opts['print_freq'] == 0):
                # Mean of accumulated values
                for k in accum.keys():
                    if k == 'length':
                        continue
                    accum[k] /= accum['length']

                print("[%s] Epoch: %d, Step: %d, Loss: %f, Poly Loss: %f,  Class Loss: %f" % (
                str(datetime.now()), epoch, self.global_step, accum['loss'], accum['poly_loss'], accum['class_loss']))
                accum = defaultdict(float)

            self.global_step += 1
        avg_epoch_loss = 0.0
        avg_poly_loss1 = 0.0
        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
            avg_poly_loss1 += poly_losses[i]

        avg_epoch_loss = avg_epoch_loss / len(losses)
        avg_poly_loss1 = avg_poly_loss1 / len(losses)
        self.gcn_loss_sum_train = avg_epoch_loss

        print("Average Epoch %d loss is : %f and poly_loss is : %f" % (epoch, avg_epoch_loss, avg_poly_loss1))

    def validate(self):
        print('Validating')
        self.model.eval()
        losses = []
        gcn_losses = []
        poly_losses = []
        class_losses = []
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
                img = img.view(-1, img.shape[0], img.shape[1], 3)
                img = torch.transpose(img, 1, 3)
                img = torch.transpose(img, 2, 3)
                img = img.float()

                self.optimizer.zero_grad()

                tg2, poly_logits, class_prob = self.model(img.to(device))

                w1 = torch.tensor(data["w"]).view(-1,1).to(device).float()
                h1 = torch.tensor(data["h"]).view(-1,1).to(device).float()

                dp_poly = data['actual_gt_poly']
                dp = torch.tensor(data['actual_gt_poly'], dtype = torch.float32).to(device)

                tg2, poly_logits, class_prob = self.model(img.to(device))


                # ----------- Mask extraction for metrics -----------
                poly_logits88 = torch.sigmoid(poly_logits[0,0,:,:]).cpu().numpy()
                yy = poly_logits88 > 0.5
                yy = yy+0
                poly_logits88 = yy.astype(np.float32)
                poly_logits89 = (poly_logits88*255).astype(np.uint8)

                kernel2 = np.ones((3,3),np.uint8)
                arrs2 = cv2.morphologyEx(poly_logits89, cv2.MORPH_CLOSE, kernel2)

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


                try:
                    h_contour = contours[largest_contour]
                except:
                    h_contour = np.zeros((5,2))
                    h_contour[:,1] = [0,w1-1,w1-1,0,0]
                    h_contour[:,0] = [0,0,h1-1,h1-1,0]


                h_contour = np.squeeze(h_contour)
                h_contour = h_contour[::-1]
                h_contour = np.asarray(h_contour)


                # ----------- saving images with contours drawn -----------
                palm_leaf_pred = copy.deepcopy(data['img_orig'][0])
                h = palm_leaf_pred.shape[0]
                w = palm_leaf_pred.shape[1]

                cv2.polylines(palm_leaf_pred, np.int32([h_contour]),True,[1], thickness = 1)
                imageio.imwrite("visualization/test_enc_pred/" + str(step) + ".jpg", palm_leaf_pred, quality=100)
                try:
                    pred_mask58 = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                    cv2.fillPoly(pred_mask58, np.int32([h_contour]),[1])
                except:
                    pred_mask58 = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)


                # ----------- GT masks -----------
                poly_mask = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                poly_mask  = utils.get_poly_mask(dp.cpu().numpy()[0],poly_mask)

                edge_mask77 = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                edge_mask77  = utils.get_edge_mask(dp.cpu().numpy()[0],edge_mask77)


                n_poly = (np.sum(poly_mask)).astype(np.float32)
                
                back_mask = 1.0-poly_mask
                n_back = (np.sum(back_mask)).astype(np.float32)
                
            
                DT_mask = compute_edts_forPenalizedLoss(edge_mask77)
                DT_mask = torch.from_numpy(np.asarray(DT_mask)).to(device)
                DT_mask = DT_mask.float()

                w1,h1 = poly_logits[:,0,:,:].shape[1],poly_logits[:,0,:,:].shape[2]
                

                self.p_n2 = torch.ones([w1,h1], dtype= torch.float32).to(device)
                self.p_n2 = self.p_n2*(n_back/n_poly)

                self.poly_loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.p_n2, reduction = 'none')


                poly_mask = torch.from_numpy(np.asarray(poly_mask)).to(device)
                poly_mask = poly_mask.view(1,poly_mask.shape[0],poly_mask.shape[1]).to(device)

                poly_loss1 = self.poly_loss_fn(poly_logits[:,0,:,:], poly_mask.to(device))
                
                # ----------- Focal Loss -----------
                # pt3 = torch.exp(-poly_loss1)
                # poly_loss1 = ((1-pt3)**2) * poly_loss1
                
                poly_loss1 = poly_loss1*DT_mask
                poly_loss1 = torch.mean(poly_loss1)

                gt_label = torch.tensor(data["gt_label"]).to(device)
                
                # ----------- Classifier loss ----------- 
                self.edge_loss_fn1 = nn.CrossEntropyLoss()
                try:
                    class_loss1 = self.edge_loss_fn1(class_prob.to(device), gt_label.to(device))
                except Exception as e:
                    # print(gt_label)
                    print(e)
                    continue

                palm_leaf_pred = copy.deepcopy(data['img_orig'][0])
                
                pred_mask = (pred_mask58*255).astype(np.uint8)
                gt_mask = ((poly_mask*255)[0].cpu().numpy()).astype(np.uint8)

                iou1, accuracy1 = utils.compute_iou_and_accuracy(pred_mask, gt_mask)
                
                # ----------- Hausdorff Distance metrics
                try:
                    hd1 = utils.hd(pred_mask, gt_mask)
                    hd951 = utils.hd95(pred_mask, gt_mask)
                except:
                    hd1 = 0
                    hd951 = 0


                avg_iou += iou1
                avg_acc += hd1

                class_lab = data['label'][0]
                
                final_ious[class_lab] += iou1
                final_acc[class_lab] += accuracy1

                final_hd[class_lab] += hd1
                final_hd95[class_lab] += hd951

                testarr.append(class_lab)

                poly_loss = poly_loss1
                
                loss_sum = 200*poly_loss1
                poly_loss_sum = poly_loss1
                class_loss_sum = class_loss1

                loss = loss_sum
                losses.append(loss)
                poly_losses.append(poly_loss_sum)
                class_losses.append(class_loss_sum)

        avg_epoch_loss = 0.0
        avg_poly_loss = 0.0
        avg_class_loss = 0.0
        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
            avg_poly_loss += poly_losses[i]
            avg_class_loss += class_losses[i]

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
        avg_poly_loss = avg_poly_loss / len(losses)
        avg_class_loss = avg_class_loss / len(losses)
        avg_iou = avg_iou / len(losses)
        avg_acc = avg_acc / len(losses)
        print("IIIIIOOOOOUUUUU", avg_iou)
        print("AAAccccuracy", avg_acc)
        print("Average VAL error is : %f, Average VAL poly error is : %f, Average VAL class error is : %f" % (avg_epoch_loss,avg_poly_loss, avg_class_loss))
        self.model.train()


if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    trainer = Trainer(args, opts)
    trainer.loop()
