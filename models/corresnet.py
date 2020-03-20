import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

TAU = 0.01

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3)
        self.in_norm1 = nn.InstanceNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3)
        self.in_norm2 = nn.InstanceNorm2d(out_channel)


    def forward(self, x):
        '''
        Parameters:
        x: Tensor of image [N, in_channel, H, W]
            Image to further exploit features
        '''

        residual = x
        out = F.prelu(self.in_norm1(self.conv1(x)))
        out = self.in_norm2(self.conv2(out))
        out += residual
        out = F.prelu(out)
        return out


class CorrespodenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        vgg19 = models.vgg19_bn()
        #vgg19 = models.vgg19()

        # Extract feature maps from VGG19 relu2_2, relu3_2, relu4_2, relu5_2
        self.vgg19_relu2_2 = nn.Sequential(
            vgg19.features[:12],  #vgg19:  8, vgg19_bn : 12
            nn.Conv2d(128, 128, kernel_size=3),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.InstanceNorm2d(256)
        )
        self.vgg19_relu3_2 = nn.Sequential(
            vgg19.features[:19],  #vgg19: 13, vgg19_bn : 19
            nn.Conv2d(256, 128, kernel_size=3),
            nn.InstanceNorm2d(128),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.InstanceNorm2d(256)
        )
        self.vgg19_relu4_2 = nn.Sequential(
            vgg19.features[:32],  #vgg19: 22, vgg19_bn : 32
            nn.Conv2d(512, 256, kernel_size=3),
            nn.InstanceNorm2d(256),
            nn.ConvTranspose2d(256, 256, kernel_size=3),
            nn.InstanceNorm2d(256)
        )
        self.vgg19_relu5_2 = nn.Sequential(
            vgg19.features[:32],  #vgg19: 31, vgg19_bn : 45
            nn.ConvTranspose2d(512, 256, kernel_size=3),
            nn.InstanceNorm2d(256),
            nn.ConvTranspose2d(256, 256, kernel_size=3),
            nn.InstanceNorm2d(256)
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
        cur_frame: Tensor of image with size [1, C, H, W]
            Current frame in the cut
        ref: Tensor of image with size [1, C, H, W]
            Reference image of the cut

        -----------
        Return:
        W with size [HW]
        S with size [HW]
        '''
        
        # Vector of extracted features
        x_feature = self.feature(cur_frame)  # [HW x C]
        y_feature = self.feature(ref)        # [HW x C]
        # Normalize vector
        x_feature /= x_feature.norm(dim=0)  # [HW x C]
        y_feature /= y_feature.norm(dim=0)  # [HW x C]

        correlation_matrix = torch.mm(x_feature, y_feature.T)   # [HW x HW]

        warped_color = self.softmax(correlation_matrix / TAU)           # [HW x HW]
        warped_color = torch.mm(warped_color, y_feature.T).diagonal()   # [HW]
        confidence_map = correlation_matrix.max(dim=1).values           # [HW]

        return warped_color, confidence_map


    def feature(self, x):
        '''
        Derive features using VGG19 relu2_2, relu3_2, relu4_2, relu5_2 and several resblocks

        ----------
        Parameters:
        x: Tensor of image with size [1, C, H, W]
            Image to get features from

        ----------
        Return:
        Vector of features with size [HW x C]
        '''

        # Get feature maps using VGG19 relu2_2, relu3_2, relu4_2, relu5_2
        relu2_2 = self.vgg19_relu2_2(x)
        relu3_2 = self.vgg19_relu3_2(x)
        relu4_2 = self.vgg19_relu4_2(x)
        relu5_2 = self.vgg19_relu5_2(x)

        feature = torch.cat((relu2_2, relu3_2, relu4_2, relu5_2), 1)

        # Feed features into several resblocks
        feature = self.resblock1(feature)
        feature = self.resblock2(feature)
        feature = self.resblock3(feature)

        # Pad to keep the original size [1, C, H, W]
        H, W = x.size()[2], x.size()[3]
        H_sub, W_sub = feature.size()[2], feature.size()[3]
        delta_H, delta_W = H - H_sub, W - W_sub
        feature = nn.ReflectionPad2d((delta_W//2, delta_W - delta_W//2, delta_H//2, delta_H - delta_H//2))(feature)

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

        x_exp = (x - x.max(dims=0).values).exp()
        delta = x_exp / x_exp.sum(dim=0)
        return delta
