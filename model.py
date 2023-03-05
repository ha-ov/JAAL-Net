import torch
import torch.nn as nn
from densenet import _DenseLayer,_Transition
from cbam_attention import *
from GatedSpatialConv import *
from Resnet import BasicBlock
import numpy as np
import cv2
from da_att import PAM_Module,CAM_Module

class Decoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = Bottleblock(in_channels,out_channels)
    
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class Double_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Double_conv,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2=nn.Conv2d(in_channels,out_channels,1,dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    
    def forward(self,x):
        a=self.conv1(x)
        a=self.bn1(a)
        a=self.relu1(a)
        
        b=self.conv2(x)
        b=self.bn2(b)
        b=self.relu2(b)
        out=a+b
        
        return out

class LF_Net(nn.Module):
    def __init__(self, in_channels,n_class):
        super().__init__()
        self.layer1 = Double_conv(in_channels,32)
        self.p1=nn.AvgPool2d(2,2)
        
        self.layer2 = Double_conv(32,64)
        self.p2=nn.AvgPool2d(2,2)
        
        self.layer3 = _DenseLayer(64,64,1,1)
        self.p3=_Transition(128,128)
        
        self.layer4=_DenseLayer(128,128,1,1)
        self.p4=_Transition(256,256)
                
        self.layer5 = Double_conv(256,512)
                
        self.decode4 = Decoder(512,256)
        self.decode3 = Decoder(256,128)
        self.decode2 = Decoder(128,64)
        self.decode1 = Decoder(64,32)
        self.conv_last = nn.Conv2d(34, n_class, 1)

        self.c1 = nn.Conv2d(512, 1, 1)
        self.c2 = nn.Conv2d(256, 1, 1)
        self.c3 = nn.Conv2d(128, 1, 1)
        self.c4 = nn.Conv2d(64, 1, 1)
        self.c5 = nn.Conv2d(32, 1, 1)
        
        self.res1 = BasicBlock(512, 512, stride=1, downsample=None)
        self.d1 = nn.Conv2d(512, 256, 1)
        self.gate1 = GatedSpatialConv2d(256,256)
        
        self.res2 = BasicBlock(256, 256, stride=1, downsample=None)
        self.d2 = nn.Conv2d(256, 128, 1)
        self.gate2 = GatedSpatialConv2d(128, 128)
        
        self.res3 = BasicBlock(128, 128, stride=1, downsample=None)
        self.d3 = nn.Conv2d(128, 64, 1)
        self.gate3 = GatedSpatialConv2d(64, 64)
        
        self.res4 = BasicBlock(64, 64, stride=1, downsample=None)
        self.d4 = nn.Conv2d(64, 32, 1)
        self.gate4 = GatedSpatialConv2d(32, 1)

    def forward(self, input):
        x_size = input.size() 
        
        e1 = self.layer1(input)
        p1=self.p1(e1)
        
        e2 = self.layer2(p1)
        p2=self.p2(e2)

        e3 = self.layer3(p2)
        p3=self.p3(e3)

        e4 = self.layer4(p3)
        p4=self.p4(e4)

        im_arr = input.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()

        f=self.layer5(p4)
        c1=self.c1(f)

        d4 = self.decode4(f, e4)
        c2=self.c2(d4)
        
        d3 = self.decode3(d4, e3) 
        c3=self.c3(d3)
        
        d2 = self.decode2(d3, e2)
        c4=self.c4(d2)
        
        d1 = self.decode1(d2,e1) 
        c5=self.c5(d1)

        x1=self.res1(f)
        x1 = F.interpolate(x1, c2.size()[2:],
                           mode='bilinear', align_corners=True)
        x1=self.d1(x1)
        x1=self.gate1(x1,c2)
        
        
        x2=self.res2(x1)
        x2 = F.interpolate(x2, c3.size()[2:],
                           mode='bilinear', align_corners=True)
        x2=self.d2(x2)
        x2=self.gate2(x2,c3)
        
        
        x3=self.res3(x2)
        x3 = F.interpolate(x3, c4.size()[2:],
                           mode='bilinear', align_corners=True)
        x3=self.d3(x3)
        x3=self.gate3(x3,c4)
        
        x4=self.res4(x3)
        x4 = F.interpolate(x4, c5.size()[2:],
                           mode='bilinear', align_corners=True)
        x4=self.d4(x4)
        x4=self.gate4(x4,c5)

        cat = torch.cat((x4, canny), dim=1)
        cat=torch.cat((cat,d1),dim=1)
        cat = self.conv_last(cat)
        cat = nn.Tanh()(cat)

        return cat

class DA_Net(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(DA_Net, self).__init__()
        self.convinput= nn.Conv2d(input_nc + output_nc, ndf,3,padding=1)
        self.conv1 = nn.Conv2d(ndf, ndf, 4, 2, 1)
        self.sa = PAM_Module(ndf)
        self.sc = CAM_Module(ndf)
        self.conv2 = nn.Conv2d(ndf, 1, 4, 2, 1)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        input= self.convinput(input)
        h1 = self.conv1(input)
        h1=self.sa(h1)
        h1=self.conv2(h1)
        
        h2=self.conv1(input)
        h2=self.sc(h2)
        h2=self.conv2(h2)

        feat_sum = h1+h2
        h=self.leaky_relu(feat_sum)
        
        output = self.sigmoid(h)
        return output
         