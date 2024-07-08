import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from msam_head import MSAM

__all__ = ["CGMANet"]

class fusion_module(nn.Module):
    def __init__(self, chan_low, chan_high, chan_1):
        super().__init__()
        self.conv1x1 = nn.Conv2d(chan_low+chan_high, chan_1, 1, bias=False)
        self.basic_bl = FHModule(chan_1, dilation_type=2)
    
    def forward(self, low_l, high_l):
        low_l_up = F.interpolate(low_l,scale_factor= 2, mode='bilinear', align_corners=False)
        output = torch.cat([low_l_up, high_l],1)
        output = self.conv1x1(output)
        output = self.basic_bl(output)
        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output
class FHModule(nn.Module):
    def __init__(self, nIn, group=1, attn=False, dilation_type=1):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.dilation_type = dilation_type
        
        self.attn = attn
        int_chan = 2*nIn
        if nIn > 100:
            int_chan = nIn
 
        print('nIn:', nIn, 'int_chan', int_chan)
        self.conv_in = Conv(nIn, int_chan, 1, 1, padding=0, groups=group, bn_acti=True)

        self.dconv3x3_no_dil = Conv(int_chan , int_chan, 3, 1, padding=1,groups=int_chan, bn_acti=False)

        if self.dilation_type == 1:
            self.dilation_rate = [2, 4, 8, 16]

        if self.dilation_type == 2:
            self.dilation_rate = [1, 2, 1, 2]
        self.dconv3x3_dil_2 = Conv(int_chan, int_chan, 3, 1, padding=self.dilation_rate[0], dilation=self.dilation_rate[0], groups=int_chan, bn_acti=True)
        self.dconv3x3_dil_4 = Conv(int_chan, int_chan, 3, 1, padding=self.dilation_rate[1], dilation=self.dilation_rate[1], groups=int_chan, bn_acti=True)
        self.avgpool = nn.AvgPool2d(8, 8)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv_si = Conv(int_chan, int_chan, 3, 1, padding=1, dilation=1, groups=int_chan, bn_acti=True)
        self.conv3x3 = Conv(int_chan, nIn, 3, 1, padding=1, groups=nIn, bn_acti=False)
        
        if self.attn:
            self.conv1x1_res = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)
        
        self.bn_relu_2 = BNPReLU(nIn)
        

    def forward(self, input):
        output = self.bn_relu_1(input)
        if self.attn:
            inp = F.avg_pool2d(input, input.size()[2:])
            inp = self.conv1x1_res(inp)
            input = input * inp
        output = self.conv_in(output)
        output1 = self.dconv3x3_no_dil(output)
        output2 = self.dconv3x3_dil_2(output)
        output2_1 = output2
        output2 = output2 + output1  
        output2 = self.dconv3x3_dil_4(output2)
        output3 = self.avgpool(output)
        output3 = self.conv_si(output3)
        output3 = self.upsample(output3)
        output = output1 + output2 + output3
        output = self.conv3x3(output)
        output = self.bn_relu_2(output)

        return output + input


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class CGMANet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, groups=32, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, groups=32, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + 3)

        # CGMA Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.CGMA_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.CGMA_Block_1.add_module("FH_Module_1_" + str(i), FHModule(64, dilation_type=2))
        self.bn_prelu_2 = BNPReLU(128 + 3)

        # CGMA Block 2
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.CGMA_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.CGMA_Block_2.add_module("FH_Module_2_" + str(i),
                                        FHModule(128, attn=False)) #changelog:attn=True
        self.bn_prelu_3 = BNPReLU(256 + 3)

        self.classifier = nn.Sequential(Conv(259, classes, 1, 1, padding=0))

    def forward(self, input):

        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)
        out_conv = output0

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        # CGMA Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.CGMA_Block_1(output1_0)
        out_cgma1 = output1
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))

        # CGMA Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.CGMA_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))

        #out = self.classifier(output2_cat) #####for only Encoder training
        #out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)  #####for only Encoder training
        #return out ###### for only Encoder training
        
        return output2_cat, out_cgma1, out_conv

class msp_decoder(nn.Module):
        def __init__(self, CGMANet):
            super().__init__()
            self.net = CGMANet
            self.reduce_output2_chan = nn.Sequential(Conv(259, 128, 1, 1, padding=0))
            self.msam = MSAM()
            self.msam_sc_1 = MSAM(pool_scale=1, in_channels=64, channels=64)
            self.msam_sc_3 = MSAM(pool_scale=3, in_channels=64, channels=64)   
            self.msam1 = MSAM(in_channels=32, channels=64)
            self.msam1_sc_1 = MSAM(pool_scale=1, in_channels=32, channels=32)
            self.msam1_sc_3 = MSAM(pool_scale=3, in_channels=32, channels=32)
            self.project = nn.Conv2d(512, 19, 1, padding=0, bias=False)
           

        def forward(self, input):
            output, out_h , out_c= self.net(input)
            output = self.reduce_output2_chan(output)
            output_msam = self.msam(out_h, output)
            output_msam_1 = self.msam_sc_1(out_h, output)
            output_msam_3 = self.msam_sc_3(out_h, output)
            output1_msam = self.msam1(out_c, output)
            output1_msam_1 = self.msam1_sc_1(out_c, output)
            output1_msam_3 = self.msam1_sc_3(out_c, output)
            output = torch.cat([output, output_msam, output_msam_1, output_msam_3, output1_msam, output1_msam_1, output1_msam_3], 1)            
            output = self.project(output)
            output = F.interpolate(output, scale_factor=8, mode='bilinear', align_corners=False)
            return output
