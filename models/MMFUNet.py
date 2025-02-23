
import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
from torch import nn
from models.FMS import FMS
from models.MDAF import MDAF
from models.Fushion import Fusion
from models.Attention import GlobalBlock,LocalBlock,DWTForward,VSSBlock
from models.HFF import HFF_block
from models.WTConv import DepthwiseSeparableConvWithWTConv2d,WTConv2d
from models.DCA import DCA
from models.downwt import Down_wt
from torchstat import stat
from timm.models.layers import DropPath, trunc_normal_



class Conv2dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(4, out_channel),
            nn.GELU()
        )
        
class Conv1dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(4, out_channel),
            nn.GELU()
        )

class InvertedDepthWiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 1x1 pointwise conv
        layers.append(Conv2dGNGELU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            Conv2dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
        
class InvertedDepthWiseConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 1x1 pointwise conv
        layers.append(Conv1dGNGELU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            Conv1dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)








    
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x
    
class DepthWiseConv2d(nn.Module):    #深度可分离卷积
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1, norm_type='gn', gn_num=4):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)                    #深度卷积层
        if norm_type == 'bn': self.norm_layer = nn.BatchNorm2d(dim_in)
        elif norm_type == 'in': self.norm_layer = nn.InstanceNorm2d(dim_in)
        elif norm_type == 'gn': self.norm_layer = nn.GroupNorm(gn_num, dim_in)
        else: raise('Error norm_type')
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)     #逐点卷积层
    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))       #先深度卷积，再归一化，再逐点卷积
     



class Myconv(nn.Module):          
    def __init__(self,
                 decode_channels,
                 gn_num=4,
                 norm_type='gn', 
                 window_size=8,
              ):
        super().__init__()   
        if norm_type == 'bn': self.norm_layer = nn.BatchNorm2d(2*decode_channels)
        elif norm_type == 'in': self.norm_layer = nn.InstanceNorm2d(2*decode_channels)
        elif norm_type == 'gn': self.norm_layer = nn.GroupNorm(gn_num,2*decode_channels )
        else: raise('Error norm_type')

        self.MDAF_L = MDAF(2*decode_channels,num_heads=8,LayerNorm_type = 'WithBias')
        self.MDAF_H = MDAF(2*decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.fuseFeature = FMS(in_ch=decode_channels, out_ch=decode_channels,num_heads=8,window_size=window_size)
        #self.WF1 = WF(in_channels=2*decode_channels,decode_channels=2*decode_channels)
        self.WF1 = HFF_block(2*decode_channels,2*decode_channels,16,2*decode_channels,2*decode_channels)
       
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(3*decode_channels, 2*decode_channels, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(2*decode_channels),
                                    nn.ReLU(inplace=True),
                                    )
        self.conv_bn_relu2= nn.Sequential(
                                    nn.Conv2d(3*decode_channels, 2*decode_channels, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(2*decode_channels),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(decode_channels,2*decode_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(2*decode_channels),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_L1= nn.Sequential(
            nn.Conv2d(decode_channels,2*decode_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(2*decode_channels),
            nn.ReLU(inplace=True),
        )

        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(2*decode_channels, 2*decode_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(2*decode_channels),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H2 = nn.Sequential(
            nn.Conv2d(2*decode_channels, 2*decode_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(2*decode_channels),
            nn.ReLU(inplace=True),
        )
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.wt1 = DWTForward(J=1, mode='zero', wave='haar')
        self.wt2 = DWTForward(J=1, mode='zero', wave='haar')
        self.con= Conv(decode_channels,decode_channels*2, kernel_size=3, stride=2, dilation=1, bias=False)
    def forward(self,glb,local,x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
       # print(yL.shape)
        yL = self.outconv_bn_relu_L(yL)     #计算低频
        yH = self.outconv_bn_relu_H(yH)     #计算高频
        glb = self.MDAF_L(yL,glb)
        local = self.MDAF_H(yH,local)
        res  = self.WF1(local,glb,x)
        return self.norm_layer(res)
           
       

class Upconv(nn.Module):          
    def __init__(self,
                 decode_channels,
                 depth1,
                 depth,
                 gn_num=4,
                 norm_type='gn', 
              
              ):
        super().__init__()   
        if norm_type == 'bn': self.norm_layer = nn.BatchNorm2d(decode_channels)
        elif norm_type == 'in': self.norm_layer = nn.InstanceNorm2d(decode_channels)
        elif norm_type == 'gn': self.norm_layer = nn.GroupNorm(gn_num,decode_channels )
        else: raise('Error norm_type')
        self.WF1 = HFF_block(decode_channels,decode_channels,16,decode_channels,decode_channels)
        self.layers = nn.ModuleList([VSSBlock(decode_channels) for _ in range(depth1)])
        self.layers2 = nn.ModuleList([LocalBlock(dim=decode_channels) for _ in range(depth)])
    def forward(self,x):
        x2=x.clone()
        x1=x.clone()
        for attn in self.layers:
            x1= attn(x1) + x1
        for attn2 in self.layers2:
            x2 = attn2(x2) + x2
        res  = self.WF1(x2,x1,None)
        res=res+x
        return self.norm_layer(res)

##################################################################################################################

class MEWB1(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([VSSBlock(dim) for _ in range(depth)])
    def forward(self, x):
        for attn in self.layers:
            x = attn(x) + x
        return x
class MEWB2(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([LocalBlock(dim=dim) for _ in range(depth)])
    def forward(self, x):
        for attn in self.layers:
            x = attn(x) + x
        return x
class BasicBlock(nn.Module):
    def __init__(self, filter_in, filter_out):
        # 初始化父类 nn.Module
        super(BasicBlock, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=filter_in, out_channels=filter_out, kernel_size=3, padding=1)
        #self.conv1=DepthWiseConv2d(filter_in,filter_out)
        # 定义第一个批量归一化层
        self.bn1 = nn.BatchNorm2d(num_features=filter_out, momentum=0.1)
        # 定义 ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=filter_out, out_channels=filter_out, kernel_size=3, padding=1)
      #  self.conv2=DepthWiseConv2d(filter_in,filter_out)
        # 定义第二个批量归一化层
        self.bn2 = nn.BatchNorm2d(num_features=filter_out, momentum=0.1)

    def forward(self, x):
        # 保存输入作为残差连接
        residual = x
        # 第一层卷积 + 批量归一化 + ReLU 激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二层卷积 + 批量归一化
        out = self.conv2(out)
        out = self.bn2(out)
        # 残差连接
        out += residual
        # 最后一次 ReLU 激活
        out = self.relu(out)
        # 返回最终输出
        return out
    
    
class MMFUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, dim= [32,64,128, 256,512], depth1=[2,2,9,2],depth=[1, 2, 2, 4], mlp_ratio=4):
        super().__init__()
        self.e0 = nn.Sequential(
           Down_wt(in_c, dim[0]),
        )
        self.g1= nn.Sequential(
            DepthWiseConv2d(dim[0], dim[1]),
            Conv(dim[1],dim[1], kernel_size=3, stride=2, dilation=1, bias=False),
            MEWB1(dim[1], depth1[0]),
            nn.Conv2d(dim[1], dim[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(dim[1]),
            nn.ReLU(inplace=True), 
        )
        self.l1 = nn.Sequential(
            DepthWiseConv2d(dim[0], dim[1]),
            Conv(dim[1],dim[1], kernel_size=3, stride=2, dilation=1, bias=False),
            MEWB2(dim[1], depth[0]),
            nn.Conv2d(dim[1], dim[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(dim[1]),
            nn.ReLU(inplace=True),
        )
        self.g2= nn.Sequential(
            DepthWiseConv2d(dim[1], dim[2]),
            Conv(dim[2],dim[2], kernel_size=3, stride=2, dilation=1, bias=False),
            MEWB1(dim[2], depth1[1]),
            nn.Conv2d(dim[2], dim[2], kernel_size=1, stride=1),
            nn.BatchNorm2d(dim[2]),
            nn.ReLU(inplace=True),
        )
        self.l2= nn.Sequential(
             DepthWiseConv2d(dim[1], dim[2]),
            Conv(dim[2],dim[2], kernel_size=3, stride=2, dilation=1, bias=False),
            MEWB2(dim[2], depth[1]),
            nn.Conv2d(dim[2], dim[2], kernel_size=1, stride=1),
            nn.BatchNorm2d(dim[2]),
            nn.ReLU(inplace=True),
        )
        self.g3= nn.Sequential(
            DepthWiseConv2d(dim[2], dim[3]),
            Conv(dim[3],dim[3], kernel_size=3, stride=2, dilation=1, bias=False),
            MEWB1(dim[3], depth1[2]),
            nn.Conv2d(dim[3], dim[3], kernel_size=1, stride=1),
            nn.BatchNorm2d(dim[3]),
            nn.ReLU(inplace=True),
        )
        self.l3= nn.Sequential(
            DepthWiseConv2d(dim[2], dim[3]),
            Conv(dim[3],dim[3], kernel_size=3, stride=2, dilation=1, bias=False),
            MEWB2(dim[3], depth[2]),
            nn.Conv2d(dim[3], dim[3], kernel_size=1, stride=1),
            nn.BatchNorm2d(dim[3]),
            nn.ReLU(inplace=True),
        )
        self.g4= nn.Sequential(
            DepthWiseConv2d(dim[3], dim[4]),
            Conv(dim[4],dim[4], kernel_size=3, stride=2, dilation=1, bias=False),
            MEWB1(dim[4], depth1[3]),
            nn.Conv2d(dim[4], dim[4], kernel_size=1, stride=1),
            nn.BatchNorm2d(dim[4]),
            nn.ReLU(inplace=True),
        )


        self.l4= nn.Sequential(
            DepthWiseConv2d(dim[3], dim[4]),
            Conv(dim[4],dim[4], kernel_size=3, stride=2, dilation=1, bias=False),#下采样
            MEWB2(dim[4], depth[3]),
            nn.Conv2d(dim[4], dim[4], kernel_size=1, stride=1),
            nn.BatchNorm2d(dim[4]),
            nn.ReLU(inplace=True),
        )

        self.f1=Myconv(dim[0])
        self.f2=Myconv(dim[1])
        self.f3=Myconv(dim[2])
        self.f4=Myconv(dim[3])
        self.d4 = nn.Sequential(
            DepthWiseConv2d(dim[4], dim[3]),  #实现维度降低
            Upconv(dim[3],depth1[3],depth[3])
              
        )
        self.d3 = nn.Sequential(
             DepthWiseConv2d(dim[3], dim[2]),  
                Upconv(dim[2],depth1[2],depth[2])
        )
          
          
        self.d2 = nn.Sequential(
            DepthWiseConv2d(dim[2], dim[1])  ,
         Upconv(dim[1],depth1[1],depth[1]),
          
        )
        self.d1 = nn.Sequential(
             DepthWiseConv2d(dim[1], dim[0]),
           Upconv(dim[0],depth1[0],depth[0])
   
        )
        self.d0 = nn.Sequential(
            nn.Conv2d(dim[0], out_c, 1)
        )

        self.resblock0 = nn.Sequential(
            BasicBlock(dim[0], dim[0]),
              BasicBlock(dim[0], dim[0]),
                 BasicBlock(dim[0], dim[0]),
                     BasicBlock(dim[0], dim[0]),
        )
        self.resblock1 = nn.Sequential(
             BasicBlock(dim[1], dim[1]),
              BasicBlock(dim[1], dim[1]),
                  BasicBlock(dim[1], dim[1]),
                     BasicBlock(dim[1], dim[1]),
        )
        self.resblock2 = nn.Sequential(
             BasicBlock(dim[2], dim[2]),
               BasicBlock(dim[2], dim[2]),
                  BasicBlock(dim[2], dim[2]),
                     BasicBlock(dim[2], dim[2]),
        )
        self.resblock3 = nn.Sequential(
             BasicBlock(dim[3], dim[3]),
               BasicBlock(dim[3], dim[3]),
                  BasicBlock(dim[3], dim[3]),
                     BasicBlock(dim[3], dim[3]),
        )
        self.resblock4= nn.Sequential(
             BasicBlock(dim[4], dim[4]),
               BasicBlock(dim[4], dim[4]),
                  BasicBlock(dim[4], dim[4]),
                     BasicBlock(dim[4], dim[4]),
        )

#########################################

        self.resblock00 = nn.Sequential(
            BasicBlock(dim[0], dim[0]),
              BasicBlock(dim[0], dim[0]),
                 BasicBlock(dim[0], dim[0]),
                    BasicBlock(dim[0], dim[0]),
        )
        self.resblock01 = nn.Sequential(
            BasicBlock(dim[1], dim[1]),
              BasicBlock(dim[1], dim[1]),
                 BasicBlock(dim[1], dim[1]),
                    BasicBlock(dim[1], dim[1]),
        )
        self.resblock02 = nn.Sequential(
            BasicBlock(dim[2], dim[2]),
              BasicBlock(dim[2], dim[2]),
                 BasicBlock(dim[2], dim[2]),
                    BasicBlock(dim[2], dim[2]),
        )
        self.resblock03 = nn.Sequential(
            BasicBlock(dim[3], dim[3]),
              BasicBlock(dim[3], dim[3]),
                 BasicBlock(dim[3], dim[3]),
                    BasicBlock(dim[3], dim[3]),
        )
        self.resblock04= nn.Sequential(
            BasicBlock(dim[4], dim[4]),
              BasicBlock(dim[4], dim[4]),
                 BasicBlock(dim[4], dim[4]),
                    BasicBlock(dim[4], dim[4]),
        )

        self.fushion4=Fusion(dim[3],wave="haar")
        self.fushion3=Fusion(dim[2],wave="haar")
        self.fushion2=Fusion(dim[1],wave="haar")
        self.fushion1=Fusion(dim[0],wave="haar")
        self.DCA=DCA(features=[32,64,128,256,512])


        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear

        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """




        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)





    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x0=self.e0(x)
        g1=self.g1(x0)
        g2=self.g2(g1)
        g3=self.g3(g2)
        g4=self.g4(g3)
        l1=self.l1(x0)
        l2=self.l2(l1)
        l3=self.l3(l2)
        l4=self.l4(l3)

        x0=self.resblock0(self.e0(x))
        x1= self.resblock1(self.f1(g1,l1,x0))
        x2=self.resblock2(self.f2(g2,l2,x1))
        x3=self.resblock3(self.f3(g3,l3,x2))
        x4=self.resblock4( self.f4(g4,l4,x3))


        x0,x1,x2,x3,x4=self.DCA((x0,x1,x2,x3,x4))
        x0= self.resblock00(x0)
        x1= self.resblock01(x1)
        x2=self.resblock02(x2)
        x3=self.resblock03(x3)
        x4=self.resblock04(x4)
       
           # ------decoder------#
        out4 =self.d4(x4) 
        out4=self.fushion4(x3,out4)    #通道数和x3一样    
        
      
        out3=self.d3(out4)
        out3=self.fushion3(x2,out3) 
        out2=self.d2(out3)
        out2=self.fushion2(x1,out2)
        out1=self.d1(out2)
        out1=self.fushion1(x0,out1)
        out0 = F.interpolate(self.d0(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True)# b, out_c, h, w
        return out0



