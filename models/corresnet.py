import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import sys

TAU = 0.01

class PadConvNorm(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, tranpose=False):
        super(PadConvNorm, self).__init__()

        if tranpose:
            self.pad = nn.Identity()
            self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        else:
            self.pad = nn.ReflectionPad2d(1)
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride)
        self.norm = nn.InstanceNorm2d(out_channel)

        self.out = nn.Sequential(
            self.pad,
            self.conv,
            self.norm
        )

    def forward(self, x):
        return self.out(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()

        self.in_channels = in_channel
        self.out_channels = out_channel

        self.downchannel = PadConvNorm(in_channel, out_channel)
        self.padconvnorm1 = PadConvNorm(in_channel, out_channel)
        self.padconvnorm2 = PadConvNorm(out_channel, out_channel)

        self.prelu = nn.PReLU()


    def forward(self, x):
        '''
        Parameters:
        x: Tensor of image [N, in_channel, H, W]
            Image to further exploit features
        '''

        residual = x if self.in_channels == self.out_channels else self.downchannel(x)
        out = self.prelu(self.padconvnorm1(x))
        out = self.padconvnorm2(out)
        out += residual
        out = self.prelu(out)
        return out


class CorrespodenceNet(nn.Module):
    def __init__(self):
        super(CorrespodenceNet, self).__init__()
        
        vgg19 = models.vgg19_bn(pretrained=True)
        #vgg19 = models.vgg19(pretrained=true)
        for param in vgg19.parameters():
            param.requires_grad = False

        # Extract feature maps from VGG19 relu2_2, relu3_2, relu4_2, relu5_2
        self.vgg19_relu2_2 = nn.Sequential(
            vgg19.features[:12],  #vgg19:  8, vgg19_bn : 12
            PadConvNorm(128, 128),
            PadConvNorm(128, 256, stride=2)
        )
        self.vgg19_relu3_2 = nn.Sequential(
            vgg19.features[:19],  #vgg19: 13, vgg19_bn : 19
            PadConvNorm(256, 128),
            PadConvNorm(128, 256)
        )
        self.vgg19_relu4_2 = nn.Sequential(
            vgg19.features[:32],  #vgg19: 22, vgg19_bn : 32
            PadConvNorm(512, 256),
            PadConvNorm(256, 256, tranpose=True)
        )
        self.vgg19_relu5_2 = nn.Sequential(
            vgg19.features[:45],  #vgg19: 31, vgg19_bn : 45
            PadConvNorm(512, 256, tranpose=True),
            PadConvNorm(256, 256, tranpose=True)
        )
        
        # Several resblocks to further exploit features
        self.resblock1 = ResBlock(256*4, 256)
        self.resblock2 = ResBlock(256, 256)
        self.resblock3 = ResBlock(256, 256)


    def forward(self, cur_frame, ref):
        '''
        Compute warped color W and confidence map S between cur_frame and ref

        -----------
        Paramaeters:
        cur_frame: Tensor of image with size [1, H, W] (L-channel of CIELAB image)
            Current frame in the cut
        ref: Tensor of image with size [3, H, W] (Lab-channel of CIELAB image)
            Reference image of the cut

        -----------
        Return:
        W with size [2 x H x W]
        S with size [H x W]
        '''

        h, w = ref.size()[1], ref.size()[2]
        
        # Vector of extracted features
        x_feature = self.feature(cur_frame) # [HW x C]
        y_feature = self.feature(ref[0].unsqueeze(0))    # [HW x C]
        # Normalize vector
        x_feature -= x_feature.mean(dim=0, keepdim=True)
        x_feature /= x_feature.norm(dim=0, keepdim=True)  # [HW x C]
        y_feature -= y_feature.mean(dim=0, keepdim=True)
        y_feature /= y_feature.norm(dim=0, keepdim=True)  # [HW x C]

        correlation_matrix = torch.mm(x_feature, y_feature.T)   # [HW x HW]

        warped_color = self.softmax(correlation_matrix / TAU)                # [HW x HW]
        warped_color = torch.mm(warped_color, ref[1:].reshape((2, -1)).T)    # [HW x 2]
        confidence_map = correlation_matrix.max(dim=1).values                # [HW]

        return warped_color.T.reshape((2, h, w)), confidence_map.reshape((h, w))


    def feature(self, image):
        '''
        Derive features using VGG19 relu2_2, relu3_2, relu4_2, relu5_2 and several resblocks

        ----------
        Parameters:
        image: Tensor of image with size [1, H, W] (L-channel of CIELAB image)
            Image to get features from

        ----------
        Return:
        Vector of features with size [HW x C]
        '''

        x = torch.cat((image, image, image), 0)
        x = x.unsqueeze(0)

        # Get feature maps using VGG19 relu2_2, relu3_2, relu4_2, relu5_2
        relu2_2 = self.vgg19_relu2_2(x)
        relu3_2 = self.vgg19_relu3_2(x)
        relu4_2 = self.vgg19_relu4_2(x)
        relu5_2 = self.vgg19_relu5_2(x)

        feature = torch.cat((relu2_2, relu3_2, relu4_2, relu5_2), 1)
        feature = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)(feature)
        
        # Feed features into several resblocks
        feature = self.resblock1(feature)
        feature = self.resblock2(feature)
        feature = self.resblock3(feature)

        return feature.reshape((feature.size()[1], -1)).T

    
    def softmax(self, x):
        '''
        Compute stable softmax over rows of x

        ----------
        Parameters:
        x: Tensor with size [H, W]

        ----------
        Return:
        Tensor with size [H, W]
        '''

        x_exp = (x - x.max(dim=1, keepdim=True).values).exp()
        delta = x_exp / x_exp.sum(dim=1, keepdim=True)
        return delta


if __name__ == '__main__':
    import cv2
    import torchvision
    import numpy as np

    # Load images
    img1 = cv2.imread('data/train/0/frames/0_0.jpg')
    img2 = cv2.imread('data/train/0/frames/0_1.jpg')
    # Resize to 64 x 64
    img1 = cv2.resize(img1, (64, 64))
    img2 = cv2.resize(img2, (64, 64))
    # Convert to CIELAB color space
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    
    # Convert to Cuda Tensor
    img1 = torchvision.transforms.ToTensor()(img1)#.to('cuda')
    img2 = torchvision.transforms.ToTensor()(img2)#.to('cuda')

    img_l = img1[0].unsqueeze(0)

    # Get the result
    net = CorrespodenceNet()#.to('cuda')
    W, S = net(img_l, img2)
    img = torch.cat((img_l, W), 0).permute(1, 2, 0)
    # Convert back to BGR image for visualizing
    img = 255*img.detach().numpy()
    img = img.astype(np.uint8)  # OpenCV supports uint8 for integer values
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # Visualize
    cv2.imshow('abc', img)
    cv2.waitKey(0)