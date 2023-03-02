import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoMap(nn.Module):
    def __init__(self, mode):
        super(AutoMap, self).__init__()
        self.mode = mode
        self.m1, self.m2 = 64, 64
        # IFFT part

        self.conv_2to1 = nn.Conv2d(2, 1, 1)
        # self.weight_2to1 = self.conv_2to1.weight.data
        self.conv_1to1_low = nn.Conv2d(1, 1, 1)
        # self.weight_1to1_low = self.conv_1to1_low.weight.data
        self.conv_1to1_high = nn.Conv2d(1, 1, 1)
        self.weight_1to1_high = self.conv_1to1_high.weight.data

        self.conv_1tom1 = nn.Conv2d(1, self.m1, 5, padding=2)
        # self.weight_1tom1 = self.conv_1tom1.weight.data
        self.conv_m1tom2 = nn.Conv2d(self.m1, self.m2, 5, padding=2)
        # self.weight_m1tom2 = self.conv_m1tom2.weight.data

        # self.conv_trasnpose_m2tom1 = nn.ConvTranspose2d(self.m2, self.m1, 5, padding=2)
        # self.conv_transpose_m1to1 = nn.ConvTranspose2d(self.m1, 1, 5, padding=2)
        self.conv_transpose_m2to1 = nn.ConvTranspose2d(self.m2, 1, 7, padding=3)
        # self.weight_transpose_m2to1 = self.conv_transpose_m2to1.weight.data
        self.conv_transpose_m2to2 = nn.ConvTranspose2d(self.m2, 2, 7, padding=3)
        # FFT part

        # self.conv_1tom1 = nn.Conv2d(1, self.m1, 5, padding=2)
        # self.conv_m1tom2 = nn.Conv2d(self.m1, self.m2, 5, padding=2)
        self.conv_1tom2 = nn.Conv2d(1, self.m2, 7, padding=3)
        # self.conv_1tom2.weight = nn.Parameter(torch.inverse(self.weight_transpose_m2to1))
        self.conv_2tom2 = nn.Conv2d(2, self.m2, 7, padding=3)

        self.conv_transpose_m2tom1 = nn.ConvTranspose2d(self.m2, self.m1, 5, padding=2)
        # self.conv_transpose_m2tom1.weight = nn.Parameter(torch.inverse(self.weight_m1tom2))
        self.conv_transpose_m1to1 = nn.ConvTranspose2d(self.m1, 1, 5, padding=2)
        # self.conv_transpose_m1to1.weight = nn.Parameter(torch.inverse(self.weight_1tom1))


        self.conv_transpose_1to1_high = nn.ConvTranspose2d(1, 1, 1)
        # self.conv_transpose_1to1_high.weight = nn.Parameter(torch.inverse(self.weight_1to1_high))
        self.conv_transpose_1to1_low = nn.ConvTranspose2d(1, 1, 1)
        # self.conv_transpose_1to1_low.weight = nn.Parameter(torch.inverse(self.weight_1to1_low))
        self.conv_transpose_1to2 = nn.ConvTranspose2d(1, 2, 1)
        # self.conv_transpose_1to2.weight = nn.Parameter(torch.inverse(self.weight_2to1))


    def forward(self, x:torch.Tensor):
        if self.mode == 'ifft':

            x = x.transpose(-1, -3).squeeze()

            x = self.conv_2to1(x)
            x = self.conv_1to1_low(x)
            x = self.conv_1to1_high(x)

            x = F.relu(self.conv_1tom1(x))
            x = F.relu(self.conv_m1tom2(x))

            x = self.conv_transpose_m2to2(x)

            x = x.transpose(-1, -3).unsqueeze(0)

        else:
            x = x.transpose(-1, -3).squeeze()

            x = F.relu(self.conv_2tom2(x))

            x = self.conv_transpose_m2tom1(x)
            x = self.conv_transpose_m1to1(x)

            x = self.conv_transpose_1to1_high(x)
            x = self.conv_transpose_1to1_low(x)
            x = self.conv_transpose_1to2(x)

            x = x.transpose(-1, -3).unsqueeze(0)

        return x