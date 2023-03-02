# https://github.com/Andy-zhujunwen/UNET-ZOO

import torch
import torch.nn as nn

class convBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(convBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class upConv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(upConv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class AttentionUnet(nn.Module):
    def __init__(self, input_ch=2, output_ch=2):
        super(AttentionUnet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = convBlock(ch_in=input_ch, ch_out=18)
        self.Conv2 = convBlock(ch_in=18, ch_out=36)
        self.Conv3 = convBlock(ch_in=36, ch_out=72)
        self.Conv4 = convBlock(ch_in=72, ch_out=144)
        self.Conv5 = convBlock(ch_in=144, ch_out=288)

        self.Up5 = upConv(ch_in=288, ch_out=144)
        self.Att5 = AttentionBlock(F_g=144, F_l=144, F_int=72)
        self.Up_conv5 = convBlock(ch_in=288, ch_out=144)

        self.Up4 = upConv(ch_in=144, ch_out=72)
        self.Att4 = AttentionBlock(F_g=72, F_l=72, F_int=36)
        self.Up_conv4 = convBlock(ch_in=144, ch_out=72)

        self.Up3 = upConv(ch_in=72, ch_out=36)
        self.Att3 = AttentionBlock(F_g=36, F_l=36, F_int=18)
        self.Up_conv3 = convBlock(ch_in=72, ch_out=36)

        self.Up2 = upConv(ch_in=36, ch_out=18)
        self.Att2 = AttentionBlock(F_g=18, F_l=18, F_int=32)
        self.Up_conv2 = convBlock(ch_in=36, ch_out=18)

        self.Conv_1x1 = nn.Conv2d(18, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)

        return d1
