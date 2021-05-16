import torch
import torch.nn as nn
import Utils.utils_gnn as utils
import torch.nn.functional as F
from .GCN import GCN
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolyGNN(nn.Module):
    def __init__(self,
                 state_dim=120,
                 n_adj=4,
                 cnn_feature_grids=None,
                 coarse_to_fine_steps=2
                 ):

        super(PolyGNN, self).__init__()

        self.state_dim = state_dim
        self.n_adj = n_adj
        self.cnn_feature_grids = cnn_feature_grids
        self.coarse_to_fine_steps = coarse_to_fine_steps

        print('Building GNN Encoder')


        fdim = 122

        self.n_points = 0
        self.feat_h1 = 0
        self.feat_w1 = 0

        self.psp_feature = [self.cnn_feature_grids[-1]]


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

    def forward(self, tg2, feature_hull, original_hull, binary_hull, bbox, mode):
        feature_hull = feature_hull.cpu()
        feature_hull = feature_hull.numpy()
        original_hull = original_hull.cpu()
        original_hull = original_hull.numpy()
        binary_hull = binary_hull.cpu()
        binary_hull = binary_hull.numpy()

        self.feat_h1, self.feat_w1 = tg2[:,0,:,:].shape[1],tg2[:,0,:,:].shape[2]

        self.cnn_feature_grids = [1, 120, self.feat_h1, self.feat_w1]
        self.psp_feature = [self.feat_w1]

        bbox = bbox.tolist()

        out_dict = {}
        hull_init_feature = torch.from_numpy(np.asarray(feature_hull))
        hull_original = torch.from_numpy(np.asarray(original_hull))
        hull_binary = torch.from_numpy(np.asarray(binary_hull))

        conv_layers = tg2.permute(0, 2, 3, 1).view(-1, 1, self.feat_h1*self.feat_w1, 120)
        conv_layers = conv_layers.to(device)

        out_dict['pred_polys'] = []

        self.n_points = hull_binary.shape[1]
        out_dict['n_points'] = self.n_points

        out_dict['hull_original'] = hull_original
        out_dict['hull_binary'] = hull_binary
        
        h = bbox[0][3]
        w = bbox[0][2]

        hull_original = hull_original.float().to(device)
        hull_binary = (hull_binary.float().to(device))

        for i in range(self.coarse_to_fine_steps):
            if i == 0:
                component = utils.prepare_gcn_component(hull_init_feature.numpy(),
                                                        self.psp_feature,
                                                        hull_init_feature.size()[1],
                                                        n_adj=20)
                hull_original = hull_original.float().to(device)
                adjacent = component['adj_matrix'].to(device)

                init_poly_idx = component['feature_indexs'].to(device)

                cnn_feature = self.sampling(init_poly_idx,
                                                        conv_layers)
                input_feature = torch.cat((cnn_feature, hull_binary), 2)
                input_feature = input_feature.to(device)


            else:
                hull_binary = gcn_pred_poly
                cnn_feature = self.interpolated_sum(conv_layers,  hull_binary, self.psp_feature, bbox)
                input_feature = torch.cat((cnn_feature, hull_binary), 2)

            gcn_pred = self.gnn[i].forward(input_feature, adjacent)
            
            gcn_pred_poly = self.add_shift(hull_binary.to(device), gcn_pred, bbox)

            out_dict['pred_polys'].append(gcn_pred_poly)
        out_dict['adjacent'] = adjacent
        return out_dict

    def add_shift(self, hull, shift, bbox):
        final = hull + shift
        return final

    def sampling(self,ids, features):
        cnn_out_feature = []
        for i in range(ids.size()[0]):
            id1 = ids[i, :, :]
            cnn_out = utils.gather_feature(id1, features[i])
            cnn_out_feature.append(cnn_out)

        concat_features = torch.stack(cnn_out_feature)
        concat_features = concat_features.view(-1, self.n_points, 120)

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

            Xs = X[i] * self.feat_h1
            X0 = torch.floor(Xs)
            X1 = X0 + 1

            Ys = Y[i] * self.feat_w1
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


            M_00 = utils.gather_feature(N1_id, cnns[i].view(1, self.feat_h1*self.feat_w1, 120))
            M_01 = utils.gather_feature(N2_id, cnns[i].view(1, self.feat_h1*self.feat_w1, 120))
            M_10 = utils.gather_feature(N3_id, cnns[i].view(1, self.feat_h1*self.feat_w1, 120))
            M_11 = utils.gather_feature(N4_id, cnns[i].view(1, self.feat_h1*self.feat_w1, 120))


            cnn_out = w_00.unsqueeze(1) * M_00 + \
                      w_01.unsqueeze(1) * M_01 + \
                      w_10.unsqueeze(1) * M_10 + \
                      w_11.unsqueeze(1) * M_11

            cnn_outs.append(cnn_out)
        cnn_outs = torch.stack(cnn_outs)
        cnn_outs = cnn_outs.view(-1, self.n_points, 120)

        return cnn_outs



    def reload(self, path, strict=False):
        print("Reloading full model from: ", path)
        self.load_state_dict(torch.load(path)['state_dict'], strict=strict)

