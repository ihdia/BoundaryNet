import torch
import torch.nn as nn
from .GCN_layer  import GraphConvolution
from .GCN_res_layer import GraphResConvolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(nn.Module):
    def __init__(self,
                 state_dim=256,
                 feature_dim=256):

        super(GCN, self).__init__()
        self.state_dim = 120

        self.gcn_0 = GraphConvolution(feature_dim, 'gcn_0', out_state_dim=self.state_dim)
        self.gcn_res_1 = GraphResConvolution(self.state_dim, 'gcn_res_1')
        self.gcn_res_2 = GraphResConvolution(self.state_dim, 'gcn_res_2')
        self.gcn_res_3 = GraphResConvolution(self.state_dim, 'gcn_res_3')
        self.gcn_res_4 = GraphResConvolution(self.state_dim, 'gcn_res_4')
        self.gcn_res_5 = GraphResConvolution(self.state_dim, 'gcn_res_5')
        self.gcn_res_6 = GraphResConvolution(self.state_dim, 'gcn_res_6')
        self.gcn_7 = GraphConvolution(self.state_dim , 'gcn_7', out_state_dim=32)

        self.fc = nn.Linear(
            in_features=32,
            out_features=2,
        )

    def forward(self, input, adj):
        input = self.gcn_0(input, adj)
        input = self.gcn_res_1(input, adj)
        input = self.gcn_res_2(input, adj)
        input = self.gcn_res_3(input, adj)
        input = self.gcn_res_4(input, adj)
        input = self.gcn_res_5(input, adj)
        input = self.gcn_res_6(input, adj)
        output = self.gcn_7(input, adj)
        return self.fc(output)