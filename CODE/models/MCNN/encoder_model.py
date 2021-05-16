import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x



class up_conv1(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv1,self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class Model(nn.Module):
    def __init__(self, input_height, input_width, input_channels):
        super(Model, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels


        # ----------- RESNET initialization  -----------
        self.layer_d0 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, stride=2, bias=True),
                                      nn.ReLU()).to(device)

        self.layer_d10 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1, stride=1, bias=True),
                                      nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1, stride=1, bias=True)).to(device)
        self.layer_d10s = nn.Conv2d(8, 16, 1, stride=1).to(device)
        self.relu_d10 = nn.ReLU().to(device)



        self.layer_d12 = nn.Sequential(nn.Conv2d(16, 32,3, padding=1, stride=1, bias=True),
                                      nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1, stride=1, bias=True)).to(device)
        self.layer_d12s = nn.Conv2d(16, 32, 1, stride=1).to(device)
        self.relu_d12 = nn.ReLU().to(device)



        self.layer_d14 = nn.Sequential(nn.Conv2d(32, 64, 3,padding=1, stride=1, bias=True),
                                      nn.ReLU(), nn.Conv2d(64, 64, 3,padding=1, stride=1, bias=True)).to(device)
        self.layer_d14s = nn.Conv2d(32, 64, 1, stride=1).to(device)
        self.relu_d14 = nn.ReLU().to(device)



        self.layer_d15 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=1, bias=True),
                                      nn.ReLU(), nn.Conv2d(128, 128, 3,padding=1,stride=1, bias=True)).to(device)
        self.layer_d15s = nn.Conv2d(64, 128, 1, stride=1).to(device)
        self.relu_d15 = nn.ReLU().to(device)


        self.adapt_avg = nn.AdaptiveMaxPool2d((2,2)).to(device)


        # ----------- SAG initialization -----------

        self.Up5 = up_conv1(ch_in=128,ch_out=64).to(device)
        self.Att5 = Attention_block(F_g=64,F_l=64,F_int=32).to(device)
        self.Up_conv5 = conv_block(ch_in=128, ch_out=64).to(device)

        self.Up4 = up_conv1(ch_in=64,ch_out=32).to(device)
        self.Att4 = Attention_block(F_g=32,F_l=32,F_int=16).to(device)
        self.Up_conv4 = conv_block(ch_in=64, ch_out=32).to(device)
        
        self.Up3 = up_conv1(ch_in=32,ch_out=16).to(device)
        self.Att3 = Attention_block(F_g=16,F_l=16,F_int=8).to(device)
        self.Up_conv3 = conv_block(ch_in=32, ch_out=16).to(device)

        self.Up2 = up_conv1(ch_in=16,ch_out=8).to(device)
        self.Att2 = Attention_block(F_g=8,F_l=8,F_int=4).to(device)
        self.Up_conv2 = conv_block(ch_in=16, ch_out=8).to(device)

        # ----------- Mask Decoder -----------
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),nn.ReLU()).to(device)
        self.mask_classifier = nn.Sequential(nn.Conv2d(8, 1, 1, padding=0, bias=True)).to(device)


        # ----------- Linear layers - region classifier -----------

        self.fc1 = nn.Sequential(nn.Linear(512, 128, bias=True), nn.ReLU()).to(device)
        self.fc2 = nn.Sequential(nn.Linear(128, 64, bias=True),nn.ReLU()).to(device)
        self.fc3 = nn.Sequential(nn.Linear(64, 8, bias=True)).to(device)



    def forward(self, input_img):
        
        # resnet backbone

        output1 = self.layer_d0(input_img.to(device))

        output_d10 = self.layer_d10(output1.to(device))
        output_d10s = self.layer_d10s(output1.to(device))
        output = output_d10 + output_d10s
        output2 = self.relu_d10(output.to(device))

        output_d12 = self.layer_d12(output2.to(device))
        output_d12s = self.layer_d12s(output2.to(device))
        output = output_d12 + output_d12s
        output3 = self.relu_d12(output.to(device))

        output_d14 = self.layer_d14(output3.to(device))
        output_d14s = self.layer_d14s(output3.to(device))
        output = output_d14 + output_d14s
        output4 = self.relu_d14(output.to(device))


        output_d15 = self.layer_d15(output4.to(device))
        output_d15s = self.layer_d15s(output4.to(device))
        output = output_d15 + output_d15s
        output5 = self.relu_d15(output.to(device))

        df48 = output5

        ## SAG Blocks

        d5 = self.Up5(output5.to(device))
        output4 = self.Att5(g=d5.to(device),x=output4.to(device))
        d5 = torch.cat((output4.to(device),d5.to(device)),dim=1)        
        d5 = self.Up_conv5(d5.to(device))
        
        d4 = self.Up4(d5.to(device))
        output3 = self.Att4(g=d4.to(device),x=output3.to(device))
        d4 = torch.cat((output3.to(device),d4.to(device)),dim=1)
        d4 = self.Up_conv4(d4.to(device))

        d3 = self.Up3(d4.to(device))
        output2 = self.Att3(g=d3.to(device),x=output2.to(device))
        d3 = torch.cat((output2.to(device),d3.to(device)),dim=1)
        d3 = self.Up_conv3(d3.to(device))

        d2 = self.Up2(d3.to(device))
        output1 = self.Att2(g=d2.to(device),x=output1.to(device))
        d2 = torch.cat((output1.to(device),d2.to(device)),dim=1)
        d2 = self.Up_conv2(d2.to(device))

        df4 = torch.cat((d2, d3, d4, d5), dim = 1)

        # decoder
        deconv_1 = self.deconv1(d2.to(device))

        mask_logits = self.mask_classifier(deconv_1.to(device))


        # class branch
        class_logits = self.adapt_avg(df48.to(device))
        class_logits = class_logits.reshape(class_logits.shape[0], -1).to(device)
        class_prob = self.fc1(class_logits.to(device))
        class_prob = self.fc2(class_prob.to(device))
        class_prob = self.fc3(class_prob.to(device))


        return df4.to(device) , mask_logits.to(device), class_prob.to(device)
