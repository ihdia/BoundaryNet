import torch
import torch.nn as nn
import Utils2.utils as utils
import torch.nn.functional as F
from GNN.GCN import GCN
import numpy as np
from losses import poly_mathcing_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class PolyGNN(nn.Module):
    def __init__(self,
                 state_dim=2,
                 n_adj=4,
                 cnn_feature_grids=None,
                 coarse_to_fine_steps=0,
                 get_point_annotation=False
                 ):

        super(PolyGNN, self).__init__()

        self.state_dim = state_dim
        self.n_adj = n_adj
        self.cnn_feature_grids = cnn_feature_grids
        self.coarse_to_fine_steps = coarse_to_fine_steps
        self.get_point_annotation = get_point_annotation

        print('Building GNN Encoder')


        if get_point_annotation:
            nInputChannels = 4
        else:
            nInputChannels = 3


        fdim = 132

        self.n_points = 0
        self.feat_h1 = 0
        self.feat_w1 = 0

        self.psp_feature = [self.cnn_feature_grids[-1]]
        # print("psp", self.psp_feature)


        if self.coarse_to_fine_steps > 0:
            for step in range(self.coarse_to_fine_steps):
                if step == 0:
                    self.gnn = nn.ModuleList(
                        [GCN(state_dim=self.state_dim, feature_dim=fdim).to(device)])
                else:
                    self.gnn.append(GCN(state_dim=self.state_dim, feature_dim=fdim).to(device))
        else:

            self.gnn = GCN(state_dim=self.state_dim, feature_dim=fdim)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # m.weight.data.normal_(0.0, 0.00002)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, tg2, feature_hull, original_hull, binary_hull, bbox, dp):
        """
        pred_polys: in scale [0,1]
        """

        feature_hull = feature_hull.cpu()
        feature_hull = feature_hull.numpy()
        original_hull = original_hull.cpu()
        original_hull = original_hull.numpy()
        binary_hull = binary_hull.cpu()
        binary_hull = binary_hull.numpy()

        self.feat_h1, self.feat_w1 = tg2[:,0,:,:].shape[1],tg2[:,0,:,:].shape[2]

        self.cnn_feature_grids = [1, 130, self.feat_h1, self.feat_w1]
        self.psp_feature = [self.cnn_feature_grids[-1]]
        # print(self.psp_feature)

        bbox = bbox.tolist()

        out_dict = {}
        hull_init_feature = torch.from_numpy(np.asarray(feature_hull))
        hull_original = torch.from_numpy(np.asarray(original_hull))
        hull_binary = torch.from_numpy(np.asarray(binary_hull))

        conv_layers = tg2.permute(0, 2, 3, 1).view(-1, 1, self.feat_h1*self.feat_w1, 130)
        conv_layers = conv_layers.to(device)

        out_dict['pred_polys'] = []

        self.n_points = hull_binary.shape[1]
        out_dict['n_points'] = self.n_points

        out_dict['hull_original'] = hull_original
        out_dict['hull_binary'] = hull_binary
        
        h = bbox[0][3]
        w = bbox[0][2]


        # pnum1 = torch.tensor(15).to(device)
        # dp1 = dp.float().to(device)
        hull_binary1 = hull_binary.float().to(device)
        hull_binary44 = hull_binary.float().to(device)

        # class_prob = torch.log_softmax(class_prob,1)
        # # print(indi)
        # indi = torch.argmax(class_prob, dim=1)
        # class_prob = torch.zeros((1,8), dtype = torch.float32)
        # class_prob[0,indi[0]] = 1.0
        # # print(class_prob)
        # feat_test = class_prob.view(1,1,8).to(device)
        # feat_test = feat_test.repeat(1,self.n_points,1)
        # print(feat_class)
        # feat_class = torch.zeros((1,1,8), dtype = torch.float32)
        hull_original = hull_original.float().to(device)
        hull_binary = hull_binary.float().to(device)
        for i in range(self.coarse_to_fine_steps):
            if i == 0:
                # print("dddddddddddd",self.psp_feature)
                component = utils.prepare_gcn_component(hull_init_feature.numpy(),
                                                        self.psp_feature,
                                                        hull_init_feature.size()[1],
                                                        n_adj=8)
                # hull_init_feature = hull_init_feature.to(device)
                # print(hull_init_feature.size()[1])

                hull_original = hull_original.float().to(device)
                adjacent = component['adj_matrix'].to(device)
                
                # print(adjacent.shape)

                # init_poly_idx = component['feature_indexs'].to(device)
                # init_poly_idx = torch.clamp(init_poly_idx, 0, (self.feat_h1-1) * (self.feat_w1-1))

                # cnn_feature = self.sampling(init_poly_idx, conv_layers)
                cnn_feature = self.interpolated_sum(conv_layers, hull_original, self.psp_feature, bbox)

                # cnn_feature = torch.zeros((cnn_feature.shape[0], cnn_feature.shape[1], cnn_feature.shape[2])).to(device)

                input_feature = torch.cat((cnn_feature, hull_binary), 2)
                input_feature = input_feature.to(device)


            else:
                hull_binary = gcn_pred_poly
                hull_init_feature = self.bin_to_hull(gcn_pred_poly, bbox)
                cnn_feature = self.interpolated_sum(conv_layers, hull_init_feature, self.psp_feature, bbox)
                # hull_init_feature = self.hull_to_bin(hull_init_feature, bbox)
                # cnn_feature = torch.zeros((cnn_feature.shape[0], cnn_feature.shape[1], cnn_feature.shape[2])).to(device)
                input_feature = torch.cat((cnn_feature, hull_binary), 2)

            gcn_pred = self.gnn[i].forward(input_feature, adjacent)
            gcn_pred_poly = self.add_shift(hull_binary.to(device), gcn_pred, bbox, i)
            # gcn_pred_poly = self.add_shift(hull_original.to(device), gcn_pred, bbox)
            # hull_original = gcn_pred_poly

            out_dict['pred_polys'].append(gcn_pred_poly)
        # gt_right_order, poly_mathcing_loss_sum = poly_mathcing_loss(self.n_points,
        #                                                             gcn_pred_poly,
        #                                                             dp1,
        #                                                             'L1')
        # for i in range(self.n_points):
        #     if hull_binary44[0,i,1] > 0.85:
        #         # ct += 1
        #         indices1[i] = 0
        out_dict['adjacent'] = adjacent
        # out_dict["right_order_index"] = indices1
        # out_dict["gt_order"] = gt_right_order

        return out_dict

    def add_shift(self, hull, shift, bbox,i):
        origin_shifted_hull = self.shift_to_origin(hull, bbox)

        or_x = origin_shifted_hull[:,:,0].view(-1, self.n_points)
        or_y = origin_shifted_hull[:,:,1].view(-1, self.n_points)


        # print(origin_shifted_hull.shape)
        # greater_zero = origin_shifted_hull != 0
        # less_zero = origin_shifted_hull == 0

        # greater_zero = greater_zero.float()
        # less_zero = less_zero.float()

        # for j in range(origin_shifted_hull.shape[0]):
        #     for i in range(self.n_points):
        #         # if not (or_x[j,i] < 0):
        #         #     shift[j,i,0] = 0.0
        #         #     shift[j,i,1] = 0.0
        #         if or_y[j,i] > 0.4 or or_y[j,i] < -0.4:
        #             shift[j,i,0] = 0.0
        #             shift[j,i,1] = 0.0
                    # print(or_y[j,i])

        # final = hull - shift * greater_zero + shift * less_zero
        # print(shift[0,55])
        # # hull = self.hull_to_bin(hull,bbox)
        final = hull + shift
        # final = self.bin_to_hull(final,bbox)
        return final

    def sampling(self,ids, features):
        # print(ids.shape, features.shape)
        cnn_out_feature = []
        for i in range(ids.size()[0]):
            id1 = ids[i, :, :]
            # print("ids", id1)
            cnn_out = utils.gather_feature(id1, features[i])
            cnn_out_feature.append(cnn_out)

        # concat_features = torch.cat(cnn_out_feature, dim=2)
        concat_features = torch.stack(cnn_out_feature)
        concat_features = concat_features.view(-1, self.n_points, 130)

        return concat_features

    def shift_to_origin(self, hull, bbox):
        out = []
        X = hull[:, :, 0]
        Y = hull[:, :, 1]

        for i in range(len(bbox)):
            w = bbox[i][2]
            h = bbox[i][3]

            X_i = X[i] - 1 / 2
            Y_i = Y[i] - 1 / 2

            reduced = torch.stack((X_i, Y_i)).transpose(0, 1)
            out.append(reduced)

        out = torch.stack(out)

        return out



    def hull_to_bin(self, hull, bbox):
        out = []
        X = hull[:, :, 0]
        Y = hull[:, :, 1]

        for i in range(len(bbox)):
            w = bbox[i][2]
            h = bbox[i][3]

            X_i = X[i] / h
            Y_i = Y[i] / w

            reduced = torch.stack((X_i, Y_i)).transpose(0, 1)
            out.append(reduced)

        out = torch.stack(out)

        return out

    def bin_to_hull(self, hull, bbox):
        out = []
        X = hull[:, :, 0]
        Y = hull[:, :, 1]

        for i in range(len(bbox)):
            w = bbox[i][2]
            h = bbox[i][3]

            X_i = X[i] * h
            Y_i = Y[i] * w

            reduced = torch.stack((X_i, Y_i)).transpose(0, 1)
            # print(reduced)
            out.append(reduced)

        out = torch.stack(out)

        return out


    def interpolated_sum(self, cnns, coords, grids, bbox):

        X = coords[:, :, 0]
        Y = coords[:, :, 1]
        cnn_outs = []

        for i in range(len(bbox)):
            w = bbox[i][2]
            h = bbox[i][3]

            Xs = X[i] / h * self.feat_h1
            X0 = torch.floor(Xs)
            X1 = X0 + 1

            Ys = Y[i] / w * self.feat_w1
            Y0 = torch.floor(Ys)
            Y1 = Y0 + 1

            w_00 = (X1 - Xs) * (Y1 - Ys)
            w_01 = (X1 - Xs) * (Ys - Y0)
            w_10 = (Xs - X0) * (Y1 - Ys)
            w_11 = (Xs - X0) * (Ys - Y0)

            X0 = torch.clamp(X0, 0, self.feat_h1 - 1)
            X1 = torch.clamp(X1, 0, self.feat_h1 - 1)
            Y0 = torch.clamp(Y0, 0, self.feat_w1 - 1)
            Y1 = torch.clamp(Y1, 0, self.feat_w1 - 1)

            N1_id = X0 * self.feat_w1 + Y0
            N2_id = X0 * self.feat_w1 + Y1
            N3_id = X1 * self.feat_w1 + Y0
            N4_id = X1 * self.feat_w1 + Y1

            N1_id = N1_id.view(-1, self.n_points)
            N2_id = N2_id.view(-1, self.n_points)
            N3_id = N3_id.view(-1, self.n_points)
            N4_id = N4_id.view(-1, self.n_points)


            M_00 = utils.gather_feature(N1_id, cnns[i].view(1, self.feat_h1*self.feat_w1, 130))
            M_01 = utils.gather_feature(N2_id, cnns[i].view(1, self.feat_h1*self.feat_w1, 130))
            M_10 = utils.gather_feature(N3_id, cnns[i].view(1, self.feat_h1*self.feat_w1, 130))
            M_11 = utils.gather_feature(N4_id, cnns[i].view(1, self.feat_h1*self.feat_w1, 130))


            cnn_out = w_00.unsqueeze(1) * M_00 + \
                      w_01.unsqueeze(1) * M_01 + \
                      w_10.unsqueeze(1) * M_10 + \
                      w_11.unsqueeze(1) * M_11

            cnn_outs.append(cnn_out)
        cnn_outs = torch.stack(cnn_outs)
        cnn_outs = cnn_outs.view(-1, self.n_points, 130)

        return cnn_outs



    def reload(self, path, strict=False):
        print("Reloading full model from: ", path)
        self.load_state_dict(torch.load(path)['state_dict'], strict=strict)

