import torch
import torch.nn.functional as F
import torch.nn as nn
class Bconv(nn.Module):
    def __init__(self,ch_in,ch_out,k,s):
        '''
        :param ch_in: 输入通道数
        :param ch_out: 输出通道数
        :param k: 卷积核尺寸
        :param s: 步长
        :return:
        '''
        super(Bconv, self).__init__()
        self.conv=nn.Conv2d(ch_in,ch_out,k,s,padding=k//2)
        self.bn=nn.BatchNorm2d(ch_out)
        self.act=nn.SiLU()
    def forward(self,x):
        '''
        :param x: 输入
        :return:
        '''
        return self.act(self.bn(self.conv(x)))
class SppCSPC(nn.Module):
    def __init__(self,ch_in,ch_out):
        '''
        :param ch_in: 输入通道
        :param ch_out: 输出通道
        '''
        super(SppCSPC, self).__init__()
        #分支一
        self.conv1=nn.Sequential(
            Bconv(ch_in,ch_out,1,1),
            Bconv(ch_out,ch_out,3,1),
            Bconv(ch_out,ch_out,1,1)
        )
        #分支二（SPP）
        self.mp1=nn.MaxPool2d(5,1,5//2) #卷积核为5的池化
        self.mp2=nn.MaxPool2d(9,1,9//2) #卷积核为9的池化
        self.mp3=nn.MaxPool2d(13,1,13//2) #卷积核为13的池化

        #concat之后的卷积
        self.conv1_2=nn.Sequential(
            Bconv(4*ch_out,ch_out,1,1),
            Bconv(ch_out,ch_out,3,1)
        )


        #分支三
        self.conv3=Bconv(ch_in,ch_out,1,1)

        #此模块最后一层卷积
        self.conv4=Bconv(2*ch_out,ch_out,1,1)
    def forward(self,x):
        #分支一输出
        output1=self.conv1(x)

        #分支二池化层的各个输出
        mp_output1=self.mp1(output1)
        mp_output2=self.mp2(output1)
        mp_output3=self.mp3(output1)

        #合并以上并进行卷积
        result1=self.conv1_2(torch.cat((output1,mp_output1,mp_output2,mp_output3),dim=1))

        #分支三
        result2=self.conv3(x)

        return self.conv4(torch.cat((result1,result2),dim=1))



##############################################
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
