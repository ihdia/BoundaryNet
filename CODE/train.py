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
from losses.Hausdorff_loss import AveragedHausdorffLoss
import warnings
from losses.fm_maps import compute_edts_forPenalizedLoss


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

        self.model_path = 'checkpoints/Final.pth'

        self.hausdorff_loss = AveragedHausdorffLoss()
        self.model = Model(self.opts)
        self.model.load_state_dict(torch.load(self.model_path)["gcn_state_dict"])


        self.train_loader, self.val_loader, self.label_to_count = get_data_loaders(self.opts['dataset'], image_provider.DataProvider)
        
        # ----------- Classes weights computation -----------
        c1 = self.label_to_count["Hole(Physical)"]
        c2 = self.label_to_count["Character Line Segment"]
        c3 = self.label_to_count["Character Component"]
        c4 = self.label_to_count["Picture"]
        c5 = self.label_to_count["Decorator"]
        c6 = self.label_to_count["Library Marker"]
        c8 = self.label_to_count["Boundary Line"]
        c9 = self.label_to_count["Physical Degradation"]
        c10 = c1 + c2 + c3 + c4 + c5 + c6 + c8 + c9
        for i in range(len(self.label_to_count)):
            c10 = c5/c1
            c20 = c5/c2
            c30 = c5/c3
            c40 = c5/c4
            c50 = c5/c5
            c60 = c5/c6
            c80 = c5/c8
            c90 = c5/c9

        self.class_weight = torch.tensor([c10,c20,c30,c40,c50,c60,c80,c90]).to(device)

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

        ct = 0
        for name, p in self.model.named_parameters():
            ct += 1
            if not p.requires_grad:
                # No optimization for frozen params
                continue
            # print(ct, p)
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

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints', 'epoch_combined_%d_step%d.pth' \
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
        self.lr_decay.load_state_dict(save_state['lr_decay'])
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

        for step, data in enumerate(self.train_loader):
            if self.global_step % self.opts['val_freq'] == 0:
                self.validate()
                self.save_checkpoint(epoch)

            img = data['img']
            img = torch.cat(img)
            img = img.view(-1, img.shape[0], img.shape[1], 3)
            img = torch.transpose(img, 1, 3)
            img = torch.transpose(img, 2, 3)
            img = img.float()


            # ----------- GT formation -----------
            bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device)
            
            w_img = torch.tensor(data["w"]).to(device).float()
            h_img = torch.tensor(data["h"]).to(device).float()
            dp_poly = data['actual_gt_poly']
            
            dp = data['actual_gt_poly11']

            output_dict, poly_logits, class_prob = self.model(img, bbox, 'train')

            poly_mask = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
            poly_mask  = utils.get_poly_mask(dp_poly.cpu().numpy()[0],poly_mask)

            edge_mask77 = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
            edge_mask77  = utils.get_edge_mask(dp_poly.cpu().numpy()[0],edge_mask77)

            n_poly = (np.sum(poly_mask)).astype(np.float32)
            
            back_mask = 1.0-poly_mask
            n_back = (np.sum(back_mask)).astype(np.float32)

            n_edge = (np.sum(edge_mask77)).astype(np.float32)
            w1,h1 = poly_logits[:,0,:,:].shape[1],poly_logits[:,0,:,:].shape[2]


            # ----------- Distance Maps Initialization -----------
            DT_mask = compute_edts_forPenalizedLoss(edge_mask77)
            DT_mask = torch.from_numpy(np.asarray(DT_mask)).to(device)
            DT_mask = DT_mask.float()

            
            # ----------- BCE Loss -----------
            self.p_n2 = torch.ones([w1,h1], dtype= torch.float32)
            self.p_n2 = (self.p_n2*(n_back/n_poly)).to(device)
            
            self.poly_loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.p_n2, reduction = 'none')
            


            poly_mask = torch.from_numpy(np.asarray(poly_mask)).to(device)
            poly_mask = poly_mask.view(1,poly_mask.shape[0],poly_mask.shape[1]).to(device)
            poly_loss1 = self.poly_loss_fn(poly_logits[:,0,:,:], poly_mask.to(device))

            # ----------- Focal Loss -----------
            pt3 = torch.exp(-poly_loss1)
            poly_loss1 = ((1-pt3))**2 * poly_loss1

            poly_loss1 = poly_loss1*DT_mask
            poly_loss1 = torch.mean(poly_loss1)

            gt_label = torch.tensor(data["gt_label"]).to(device)
            

            # ----------- classifier loss -----------
            self.edge_loss_fn1 = nn.CrossEntropyLoss(weight = self.class_weight)
            # try:
            #     class_loss1 = self.edge_loss_fn1(class_prob.to(device), gt_label.to(device))
            # except Exception as e:
            #     # print(gt_label)
            #     print(e)
            #     continue
            class_loss1 = 0


            pred_cps = (output_dict['pred_polys'][-1]).float()
            pred_cps88 = pred_cps


            n_points = output_dict['n_points']

            hull_binary = output_dict['hull_binary'].float().to(device)
            hull_original = output_dict['hull_original'].float().to(device)


            # ----------- Graph Interpolation -----------

            # B, N_old, N_new = pred_cps.shape[0], pred_cps.shape[1], pred_cps.shape[1] * 10
            # grid = torch.cat([torch.zeros((B, N_new, 1, 1)),
            #                   torch.linspace(-1, 1, N_new).unsqueeze(0).repeat((B, 1)).unsqueeze(
            #                       -1).unsqueeze(-1)], dim=-1).to(device)

            # rows = F.grid_sample(pred_cps88.unsqueeze(1)[:, :, :, 0].unsqueeze(-1), grid).squeeze(1)
            # cols = F.grid_sample(pred_cps88.unsqueeze(1)[:, :, :, 1].unsqueeze(-1), grid).squeeze(1)
            # pred_cps88 = torch.cat([rows,cols], dim=-1)

            # n_points = 10*n_points

            pred_cps5 = pred_cps88.detach().cpu().numpy()

            # ----------- Uniform Sampling of target points on GT contour -----------
            dp = utils.uniformsample_batch(dp, n_points)
            dp = dp[0]
            dp = dp.cpu().numpy()
            dp = np.asarray(dp)
            

            x = dp[:, 1]/float(h_img[0])
            y = dp[:, 0]/float(w_img[0])

            dp_x = x
            dp_y = y

            dp_x = (torch.from_numpy(dp_x)).view(-1, n_points, 1)
            dp_y = (torch.from_numpy(dp_y)).view(-1, n_points, 1)
            dp = torch.cat((dp_x, dp_y), dim=2).to(device)


            dpf = dp.cpu().numpy()
            dpfer = dp.cpu().numpy()

            # ----------- Hausdorff Loss -----------
            han_loss = self.hausdorff_loss(pred_cps88[0,:,:].float(), dp[0,:,:].float())

            if self.opts["enc_freeze"]:
                loss_v = han_loss
            else:
                loss_v = han_loss + 200*poly_loss1

            loss_sum = han_loss
            self.gcn_loss_sum_train = han_loss
            poly_loss_sum = poly_loss1
            class_loss_sum = class_loss1

            self.optimizer.zero_grad()
            # self.optimizer1.zero_grad()

            loss_v.backward()

            if 'grad_clip' in self.opts.keys():
                nn.utils.clip_grad_norm_(self.model.parameters(), self.opts['grad_clip'])

            self.optimizer.step()
            # self.optimizer1.step()

            loss = loss_sum

            losses.append(loss_sum)
            gcn_losses.append(self.gcn_loss_sum_train)
            poly_losses.append(poly_loss_sum)
            class_losses.append(class_loss_sum)



            accum['loss'] += float(loss)
            accum['gcn_loss'] += float(self.gcn_loss_sum_train)
            accum['edge_loss'] += float(poly_loss_sum)
            accum['vertex_loss'] += float(class_loss_sum)
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

    def validate(self):
        print('Validating')
        self.model.eval()
        losses = []
        gcn_losses = []
        poly_losses = []
        urls = []
        class_losses = []
        pred_cm = []
        gt_cm = []
        avg_acc = 0.0
        avg_iou = 0.0
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

                self.optimizer.zero_grad()

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

                # ----------- Distance maps computation -----------
                DT_mask = compute_edts_forPenalizedLoss(edge_mask77)
                DT_mask = torch.from_numpy(np.asarray(DT_mask)).to(device)
                DT_mask = DT_mask.float()

                # ----------- BCE Loss -----------
                self.p_n2 = torch.ones([w1,h1], dtype= torch.float32)
                self.p_n2 = (self.p_n2*(n_back/n_poly)).to(device)            
                self.poly_loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.p_n2,reduction = 'none')

                poly_mask = torch.from_numpy(np.asarray(poly_mask)).to(device)
                poly_mask = poly_mask.view(1,poly_mask.shape[0],poly_mask.shape[1]).to(device)

                poly_loss1 = self.poly_loss_fn(poly_logits[:,0,:,:], poly_mask.to(device))
                
                # ----------- Focal Loss -----------
                pt3 = torch.exp(-poly_loss1)
                poly_loss1 = ((1-pt3))**2 * poly_loss1
                
                poly_loss1 = poly_loss1*DT_mask
                

                poly_loss1 = torch.mean(poly_loss1)

                self.edge_loss_fn1 = nn.CrossEntropyLoss(weight = self.class_weight)


                gt_label = torch.tensor(data["gt_label"]).to(device)


                try:
                    class_loss1 = self.edge_loss_fn1(class_prob.to(device), gt_label.to(device))
                except Exception as e:
                    # print(gt_label)
                    print(e)
                    continue

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

                han_loss = self.hausdorff_loss(pred_cps[0,:,:].float(), dp[0,:,:].float())

                loss_sum = han_loss
                self.gcn_loss_sum_val = han_loss
                poly_loss_sum = poly_loss1
                class_loss_sum = class_loss1


                loss = loss_sum
                losses.append(loss)
                gcn_losses.append(self.gcn_loss_sum_val)
                poly_losses.append(poly_loss_sum)
                class_losses.append(class_loss_sum)

        cm = sm.confusion_matrix(gt_cm, pred_cm, labels = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker','Boundary Line','Physical Degradation'])
        print(cm)

        avg_epoch_loss = 0.0
        avg_gcn_loss = 0.0
        avg_poly_loss = 0.0
        avg_class_loss = 0.0


        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
            avg_gcn_loss += gcn_losses[i]
            avg_poly_loss += poly_losses[i]
            avg_class_loss += class_losses[i]

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

        avg_epoch_loss = avg_epoch_loss / len(losses)
        avg_gcn_loss = avg_gcn_loss / len(losses)
        avg_poly_loss = avg_poly_loss / len(losses)
        avg_class_loss = avg_class_loss / len(losses)
        self.gcn_loss_sum_val = avg_gcn_loss
        avg_iou = avg_iou / len(losses)
        avg_acc = avg_acc / len(losses)
        print("Avg. IOU", avg_iou)
        print("Avg. Accuracy", avg_acc)

        print(avg_gcn_loss)


        print("Average VAL error is : %f, Average VAL gcn error is : %f, Average VAL poly error is : %f, Average VAL class error is : %f" % (avg_epoch_loss, avg_gcn_loss, avg_poly_loss, avg_class_loss))
        self.model.train()


if __name__ == '__main__':
    args = get_args()
    
    opts = json.load(open(args.exp, 'r'))
    trainer = Trainer(args, opts)
    trainer.loop()
