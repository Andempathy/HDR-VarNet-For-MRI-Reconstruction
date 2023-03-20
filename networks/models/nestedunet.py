# https://github.com/MrGiovanni/UNetPlusPlus
# https://github.com/Andy-zhujunwen/UNET-ZOO

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, 3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

# class ConvBlock(nn.Module):
#     """
#     A Convolutional Block that consists of two convolution layers each followed by
#     instance normalization, LeakyReLU activation and dropout.
#     """

#     def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
#         """
#         Args:
#             in_chans: Number of channels in the input.
#             out_chans: Number of channels in the output.
#             drop_prob: Dropout probability.
#         """
#         super().__init__()

#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.drop_prob = drop_prob

#         self.layers = nn.Sequential(
#             nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(out_chans),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Dropout2d(drop_prob),
#             nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(out_chans),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Dropout2d(drop_prob),
#         )

#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             image: Input 4D tensor of shape `(N, in_chans, H, W)`.

#         Returns:
#             Output tensor of shape `(N, out_chans, H, W)`.
#         """
#         return self.layers(image)

class NestedUnet(nn.Module):
    def __init__(self, in_chans, out_chans, chans, deepsupervision: bool=False):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(in_chans, chans)
        self.conv1_0 = ConvBlock(chans, chans*2)
        self.conv2_0 = ConvBlock(chans*2, chans*4)
        self.conv3_0 = ConvBlock(chans*4, chans*8)
        self.conv4_0 = ConvBlock(chans*8, chans*16)

        self.conv0_1 = ConvBlock(chans + chans*2, chans)
        self.conv1_1 = ConvBlock(chans*2 + chans*4, chans*2)
        self.conv2_1 = ConvBlock(chans*4 + chans*8, chans*4)
        self.conv3_1 = ConvBlock(chans*8 + chans*16, chans*8)

        self.conv0_2 = ConvBlock(chans*2 + chans*2, chans)
        self.conv1_2 = ConvBlock(chans*2*2 + chans*4, chans*2)
        self.conv2_2 = ConvBlock(chans*4*2 + chans*8, chans*4)

        self.conv0_3 = ConvBlock(chans*3 + chans*2, chans)
        self.conv1_3 = ConvBlock(chans*2*3 + chans*4, chans*2)

        self.conv0_4 = ConvBlock(chans*4 + chans*2, chans)
        self.sigmoid = nn.Sigmoid()
        if deepsupervision:
            self.final1 = nn.Conv2d(chans, out_chans, kernel_size=1)
            self.final2 = nn.Conv2d(chans, out_chans, kernel_size=1)
            self.final3 = nn.Conv2d(chans, out_chans, kernel_size=1)
            self.final4 = nn.Conv2d(chans, out_chans, kernel_size=1)
        else:
            self.final = nn.Conv2d(chans, out_chans, kernel_size=1)
        
        self.deepsupervision = deepsupervision


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output1 = self.sigmoid(output1)
            output2 = self.final2(x0_2)
            output2 = self.sigmoid(output2)
            output3 = self.final3(x0_3)
            output3 = self.sigmoid(output3)
            output4 = self.final4(x0_4)
            output4 = self.sigmoid(output4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            output = self.sigmoid(output)
            return output