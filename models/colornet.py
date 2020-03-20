import torch
import torch.nn as nn

class Colornet(nn.Module):
    """
    Colorization Net
    """

    def __init__(self):
        super(Colornet, self).__init__()
        use_bias = True
        norm_layer = nn.InstanceNorm2d

        # Conv block 1
        conv_block_1 = [nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_1 += [nn.ReLU(True), ]
        conv_block_1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_1 += [nn.ReLU(True), ]
        conv_block_1 += [norm_layer(64), ]
        conv_block_1 += [nn.MaxPool2d(2, stride=1),]


        # Conv block 2
        conv_block_2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_2 += [nn.ReLU(True), ]
        conv_block_2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_2 += [nn.ReLU(True), ]
        conv_block_2 += [norm_layer(128), ]
        conv_block_2 += [nn.MaxPool2d(2, stride=1),]

        # Conv block 3
        conv_block_3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_3 += [nn.ReLU(True), ]
        conv_block_3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_3 += [nn.ReLU(True), ]
        conv_block_3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        conv_block_3 += [nn.ReLU(True), ]
        conv_block_3 += [norm_layer(256), ]
        conv_block_3 += [nn.MaxPool2d(2, stride=1),]

        # Resblock 1
        # TODO: Add a downsample from 256 -> 512
        # TODO: Adding before taking relu, might need to read resnet paper
        # again
        res_block_1 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        res_block_1 += [nn.ReLU(True), ]
        res_block_1 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        res_block_1 += [nn.ReLU(True), ]
        res_block_1 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        res_block_1 += [nn.ReLU(True), ]

        # Resblock 2
        res_block_2 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        res_block_2 += [nn.ReLU(True), ]
        res_block_2 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        res_block_2 += [nn.ReLU(True), ]
        res_block_2 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        res_block_2 += [nn.ReLU(True), ]

        # Resblock 4
        res_block_3 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        res_block_3 += [nn.ReLU(True), ]
        res_block_3 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        res_block_3 += [nn.ReLU(True), ]
        res_block_3 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        res_block_3 += [nn.ReLU(True), ]

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
        self.conv_block_2 = nn.Sequential(*conv_block_2)
        self.conv_block_3 = nn.Sequential(*conv_block_3)

        self.res_block_1 = nn.Sequential(*res_block_1)
        self.res_block_2 = nn.Sequential(*res_block_2)
        self.res_block_3 = nn.Sequential(*res_block_3)

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
        input_stacked = torch.cat((frame_cur, frame_prev, Wab, S), 1)

        # Downscale convolution bloc
        conv1 = self.conv_block_1(input_stacked)
        conv2 = self.conv_block_2(conv1)
        conv3 = self.conv_block_3(conv2)

        # Residual block
        res1 = self.res_block_1(conv3) + conv3
        res2 = self.res_block_2(res1) + res1
        res3 = self.res_block_3(res2) + res2

        # Upscale conv blocks
        cov7_up = self.upscale_7(res3) + self.skip_1_7(conv3)
        cov7 = self.conv_block_7(cov7_up)

        cov8_up = self.upscale_8(cov7) + self.skip_2_8(conv2)
        cov8 = self.conv_block_8(cov8_up)

        cov9_up = self.upscale_7(cov8) + self.skip_1_7(conv1)
        cov9 = self.conv_block_7(cov9_up)

        output = self.conv_final(cov9)

        return output

