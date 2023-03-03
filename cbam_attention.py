import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes//2, 1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 2, in_planes, 1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return self.sigmoid(avg_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.aspp=ASPP()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out,max_out], dim=1)
        x = self.conv1(x)
        aspp=self.aspp(x) 
        x=x+aspp
        
        return self.sigmoid(x)

    
class Bottleblock(nn.Module):

    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2=nn.Conv2d(planes,planes,3,padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        
        
        self.conv3=nn.Conv2d(inplanes,planes,1,dilation=2)
        self.bn3 = nn.BatchNorm2d(planes)
        

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out=self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out1=self.conv3(x)
        out1 = self.bn3(out1)
        out1=self.relu(out1)
        
        out=out+out1

        out = self.ca(out) * out
        out = self.sa(out) * out
        
        return out


class ASPP(nn.Module):
    def __init__(self, in_channel=1, depth=1):
        super(ASPP, self).__init__()
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 2, depth, 1, 1)

    def forward(self, x):
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = atrous_block6 + atrous_block12 + atrous_block18
        return net

    
        