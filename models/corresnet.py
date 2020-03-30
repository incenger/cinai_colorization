import torch
import torch.nn as nn
import torchvision.models as models

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


class ResBlockCorresnet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockCorresnet, self).__init__()

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
            Image to further exploit features.
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
            vgg19.features[:13],  #vgg19:  8, vgg19_bn : 12
            PadConvNorm(128, 128),
            nn.ReLU(True),
            PadConvNorm(128, 256, stride=2),
            nn.ReLU(True),
        )
        self.vgg19_relu3_2 = nn.Sequential(
            vgg19.features[:20],  #vgg19: 13, vgg19_bn : 19
            PadConvNorm(256, 128),
            nn.ReLU(True),
            PadConvNorm(128, 256),
            nn.ReLU(True),
        )
        self.vgg19_relu4_2 = nn.Sequential(
            vgg19.features[:33],  #vgg19: 22, vgg19_bn : 32
            PadConvNorm(512, 256),
            nn.ReLU(True),
            PadConvNorm(256, 256, tranpose=True),
            nn.ReLU(True),
        )
        self.vgg19_relu5_2 = nn.Sequential(
            vgg19.features[:46],  #vgg19: 31, vgg19_bn : 45
            PadConvNorm(512, 256, tranpose=True),
            nn.ReLU(True),
            PadConvNorm(256, 256, tranpose=True),
            nn.ReLU(True),
        )
        
        # Several resblocks to further exploit features
        self.resblock1 = ResBlockCorresnet(256*4, 256)
        self.resblock2 = ResBlockCorresnet(256, 256)
        self.resblock3 = ResBlockCorresnet(256, 256)


    def forward(self, cur_frame, ref):
        '''
        Compute warped color W and confidence map S between cur_frame and ref.

        -----------
        Paramaeters:
        cur_frame: Tensor of image with size [1, 1, H, W]
            Current frame (L-channel in CIELAB color space) in the cut.
        ref: Tensor of image with size [1, 3, H, W]
            Reference image (Lab-channel in CIELAB color space) of the cut.

        -----------
        Return:
        W with size [1, 2, H, W] (ab-channel in CIELAB colo space)
        S with size [1, 1, H, W]
        '''

        h, w = ref.size()[2], ref.size()[3]
        
        # Vector of extracted features
        x_feature = self.feature(cur_frame) # [HW, C]
        y_feature = self.feature(ref[:, :1]) # [HW, C]
        # Normalize vector
        x_feature = (x_feature - x_feature.mean(dim=0)) / x_feature.norm(dim=0) # [HW, C]
        y_feature = (y_feature - y_feature.mean(dim=0)) / y_feature.norm(dim=0) # [HW, C]

        # Initialize W and S
        warped_color = torch.zeros((h*w, 2))#, requires_grad=True)
        confidence_map = torch.zeros(h*w)#, requires_grad=True)
        if torch.cuda.is_available():
            warped_color = warped_color.cuda()
            confidence_map = confidence_map.cuda()

        for idx in range(0, x_feature.size()[0], 2):
            correlation = torch.mm(x_feature[idx:idx+2], y_feature.T)    # [2, HW]
            warped_color[idx:idx+2] = torch.mm(nn.Softmax(dim=1)(correlation/TAU), ref[0, 1:].reshape((2, -1)).T)  # [2, 2]
            confidence_map[idx:idx+2] = correlation.max(dim=1).values

        return warped_color.T.reshape((1, 2, h, w)), confidence_map.reshape((1, 1, h, w))


    def feature(self, image):
        '''
        Derive features using VGG19 relu2_2, relu3_2, relu4_2, relu5_2 and several resblocks.

        ----------
        Parameters:
        image: Tensor of image with size [1, 1, H, W]
            Image (L-channel in CIELAB color space) to get features from.

        ----------
        Return:
        Vector of features with size [HW, C]
        '''

        x = torch.cat((image, image, image), 1)

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


if __name__ == '__main__':
    from skimage import color
    import cv2
    import torchvision
    import numpy as np

    # Load images
    img1 = cv2.imread('data/cut_000/frames/000_726.jpg')
    img2 = cv2.imread('data/cut_000/frames/000_738.jpg')
    # Resize to 64 x 64
    img1 = cv2.resize(img1, (112, 64))
    img2 = cv2.resize(img2, (112, 64))
    # Convert to CIELAB color space
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = color.rgb2lab(img1).astype(np.float32)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = color.rgb2lab(img2).astype(np.float32)
    
    # Convert to Tensor
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
    img1[0, 0] -= 50.
    img2[0, 0] -= 50.

    # Prepare model
    net = CorrespodenceNet()

    # Convert to CUDA
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
        net.cuda()

    img_l = img1[:, :1]

    # Get warped color and confidence map
    W, S = net(img_l, img2)
    
    img_l += 50.
    img = torch.cat((img_l, W), 1).squeeze().permute(1, 2, 0)
    # # Convert back to BGR image for visualizing
    img = img.detach().numpy().astype(np.float64)
    img = color.lab2rgb(img).astype(np.float32)
    img = (255*img).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    print('Done')
    
    # Visualize
    cv2.imshow('abc', img)
    cv2.waitKey(0)