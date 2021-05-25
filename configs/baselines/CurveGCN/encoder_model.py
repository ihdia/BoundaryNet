import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import random
from skimage.io import imsave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, input_height, input_width, input_channels):
        super(Model, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.layer_d0 = nn.Sequential(nn.Conv2d(3, 8, 5, padding=2, stride=2, bias=True),
                                      nn.ReLU())


        self.layer_d10 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1, stride=1, bias=True),
                                      nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1, stride=1, bias=True))
        self.layer_d10s = nn.Conv2d(8, 16, 1, stride=1)
        self.relu_d10 = nn.ReLU()

        self.layer_d11 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1, stride=2, bias=True),
                                      nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1, stride=1, bias=True))
        self.layer_d11s = nn.Conv2d(16, 32, 1, stride=2)
        self.relu_d11 = nn.ReLU()



        self.layer_d12 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=True),
                                      nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1, stride=1, bias=True))
        self.layer_d12s = nn.Conv2d(32, 64, 1, stride=2)
        self.relu_d12 = nn.ReLU()

        self.layer_d13 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=True),
                                      nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1, stride=1, bias=True))
        self.layer_d13s = nn.Conv2d(64, 128, 1, stride=2)
        self.relu_d13 = nn.ReLU()



        self.layer100 = nn.Sequential(nn.Conv2d(128, 10, 1, padding=0, bias=True), nn.ReLU())
        self.layer101 = nn.Sequential(nn.Conv2d(128, 10, 1, padding=0, bias=True), nn.ReLU())
        # self.layer102 = nn.Sequential(nn.Conv2d(180, 1, 1, padding=0, bias=True), nn.ReLU())


        self.layer9 = nn.Sequential(nn.Linear(7840, 784, bias=True))
        self.layer99 = nn.Sequential(nn.Linear(7840, 784, bias=True))

    def forward(self, input_img):

        output = self.layer_d0(input_img)


        output_d10 = self.layer_d10(output)
        output_d10s = self.layer_d10s(output)
        output = output_d10 + output_d10s
        output = self.relu_d10(output)

        output_d11 = self.layer_d11(output)
        output_d11s = self.layer_d11s(output)
        output = output_d11 + output_d11s
        output = self.relu_d11(output)

        output_d12 = self.layer_d12(output)
        output_d12s = self.layer_d12s(output)
        output = output_d12 + output_d12s
        output = self.relu_d11(output)

        output_d13 = self.layer_d13(output)
        output_d13s = self.layer_d13s(output)
        output = output_d13 + output_d13s
        output = self.relu_d13(output)


        df4 = output

        poly_logits = self.layer100(output)
        vertex_logits = self.layer101(output)
        # # edge_logits = self.layer102(output)
        # poly_logits77 = poly_logits
        # print(vertex_logits.shape)
        vertex_logits = vertex_logits.reshape(vertex_logits.shape[0], -1)
        vertex_logits = self.layer99(vertex_logits)
        # edge_logits = nn.Dropout(p=0.4)(edge_logits)
        vertex_logits1 = vertex_logits.reshape(vertex_logits.shape[0],1, 28, 28)

        poly_logits = poly_logits.reshape(poly_logits.shape[0], -1)
        poly_logits = self.layer9(poly_logits)
        # edge_logits = nn.Dropout(p=0.4)(edge_logits)
        poly_logits1 = poly_logits.reshape(poly_logits.shape[0],1, 28, 28)


        vertex_logits2 = vertex_logits1
        vertex_logits2 = torch.sigmoid(vertex_logits2)

        poly_logits2 = poly_logits1
        poly_logits2 = torch.sigmoid(poly_logits2)

        df4 = torch.cat((df4 ,poly_logits2, vertex_logits2),1)


        return df4 , vertex_logits1, poly_logits1
