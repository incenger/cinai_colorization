import torch
import torch.nn as nn

class ResBlockColornet(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResBlockColornet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, bias=True)
        self.insnom1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=True)
        self.insnom2 = nn.InstanceNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=True)
        self.insnom3 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.insnom1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.insnom2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.insnom3(out)
        if self.downsample:
            residual = self.downsample(residual)
        out = self.relu(out + residual)
        return out




class Colornet(nn.Module):
    """
    Colorization Net
    """

    def __init__(self):
        super(Colornet, self).__init__()
        use_bias = True
        norm_layer = nn.InstanceNorm2d

        # Conv block 1
        conv_block_1 = [nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_1 += [nn.ReLU(True), ]
        conv_block_1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_1 += [nn.ReLU(True), ]
        conv_block_1 += [norm_layer(64), ]
        downscale_1 = [nn.MaxPool2d(2, stride=2),]


        # Conv block 2
        conv_block_2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_2 += [nn.ReLU(True), ]
        conv_block_2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_2 += [nn.ReLU(True), ]
        conv_block_2 += [norm_layer(128), ]
        downscale_2 = [nn.MaxPool2d(2, stride=2),]

        # Conv block 3
        conv_block_3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_3 += [nn.ReLU(True), ]
        conv_block_3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_3 += [nn.ReLU(True), ]
        conv_block_3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_3 += [nn.ReLU(True), ]
        conv_block_3 += [norm_layer(256), ]
        downscale_3 = [nn.MaxPool2d(2, stride=2),]

        # Resblock 1
        downsample = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,
                                             padding=1, bias=use_bias),
                                   norm_layer(512))
        res_block_1 = ResBlockColornet(256, 512, downsample)

        # Resblock 2
        res_block_2 = ResBlockColornet(512, 512)

        # Resblock 3
        res_block_3 = ResBlockColornet(512, 512)

        # Conv block 7
        upscale_7 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding = 1, bias = use_bias)]
        skip_3_7 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias)]

        conv_block_7 = [nn.ReLU(True), ]
        conv_block_7 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_7 += [nn.ReLU(True), ]
        conv_block_7 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_7 += [nn.ReLU(True), ]
        conv_block_7 += [norm_layer(256), ]

        # Conv block 8
        upscale_8 = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding = 1, bias = use_bias)]
        skip_2_8 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias)]

        conv_block_8 = [nn.ReLU(True), ]
        conv_block_8 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_8 += [nn.ReLU(True), ]
        conv_block_8 += [norm_layer(128), ]

        # Conv block 9
        upscale_9 = [nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding = 1, bias = use_bias)]
        skip_1_9 = [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias)]

        conv_block_9 = [nn.ReLU(True), ]
        conv_block_9 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_9 += [nn.ReLU(True), ]
        conv_block_9 += [norm_layer(64), ]

        # Final bloc
        conv_final = [nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        conv_final += [nn.Tanh()]

        self.conv_block_1 = nn.Sequential(*conv_block_1)
        self.downscale_1 = nn.Sequential(*downscale_1)
        self.conv_block_2 = nn.Sequential(*conv_block_2)
        self.downscale_2 = nn.Sequential(*downscale_2)
        self.conv_block_3 = nn.Sequential(*conv_block_3)
        self.downscale_3 = nn.Sequential(*downscale_3)

        self.res_block_1 = res_block_1
        self.res_block_2 = res_block_2
        self.res_block_3 = res_block_3

        self.upscale_7 = nn.Sequential(*upscale_7)
        self.skip_3_7 = nn.Sequential(*skip_3_7)
        self.conv_block_7 = nn.Sequential(*conv_block_7)

        self.upscale_8 = nn.Sequential(*upscale_8)
        self.skip_2_8 = nn.Sequential(*skip_2_8)
        self.conv_block_8 = nn.Sequential(*conv_block_8)

        self.upscale_9 = nn.Sequential(*upscale_9)
        self.skip_1_9 = nn.Sequential(*skip_1_9)
        self.conv_block_9 = nn.Sequential(*conv_block_9)

        self.conv_final = nn.Sequential(*conv_final)


    def forward(self, frame_prev, frame_cur, Wab, S):
        """
        Parameters:
        frame_prev: tensor
            The two channels ab of the previous colorized frame
        frame_cur: tensor
            The channel l of the current frame
        Wab: tensor
            The warped color from the correspondece net
        S: tensor
            The similarity matrix from the correspondece net

        Returns:
        -------
        output: tensor
            The current frame after being colorized by the net
        """
        # Stacking four inputs into 6 channels

        # [N, C, H, W]
        input_stacked = torch.stack((frame_cur, frame_prev, Wab, S), 0)

        # Downscale convolution bloc
        conv1 = self.conv_block_1(input_stacked)
        conv2 = self.conv_block_2(self.downscale_1(conv1))
        conv3 = self.conv_block_3(self.downscale_2(conv2))

        # Residual block
        res = self.res_block_1(self.downscale_3(conv3))
        res = self.res_block_2(res)
        res = self.res_block_3(res)

        # Upscale conv blocks
        conv_up = self.upscale_7(res) + self.skip_3_7(conv3)
        conv_up = self.conv_block_7(conv_up)

        covn_up = self.upscale_8(conv_up) + self.skip_2_8(conv2)
        conv_up = self.conv_block_8(covn_up)

        conv_up = self.upscale_9(conv_up) + self.skip_1_9(conv1)
        conv_up = self.conv_block_9(conv_up)

        output = self.conv_final(conv_up)

        return output

