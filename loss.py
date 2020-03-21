import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19_bn
import  torchvision.transforms as transforms

PERCEPTUAL_LAMDA = 0.001
CONTEXTUAL_LAMDA = 0.2
SMOOTHNESS_LAMDA = 5.0
ADVERSARIAL_LAMDA = 0.2
FLOW_LOSS = 0.02
L1_LOSS = 0.1

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        # Extract vgg19 relu5_2 features
        vgg = vgg19_bn(pretrained=True)
        vgg_feature = nn.Sequential(*list(vgg.features)[:45]).eval()
        for param in vgg_feature.parameters():
            param.requires_grad = False
        self.vgg_feature = vgg_feature


    def forward(self, res, ground_truth, ref):
        perceptual_loss = (self.vgg_feature(res) - self.vgg_feature(ground_truth)).norm()
        
        return PERCEPTUAL_LAMDA*perceptual_loss


if __name__ == '__main__':
    import cv2

    img1 = cv2.imread('data/train/0/frames/0_0.jpg', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('data/train/0/frames/0_1.jpg', cv2.IMREAD_UNCHANGED)
    
    # Convert to tensor with size [1, C, H, W] from [H, W, C]
    img1 = transforms.ToTensor()(img1).unsqueeze(0)
    img2 = transforms.ToTensor()(img2).unsqueeze(0)
    #print(img1.size())

    loss = Loss()
    print(loss(img1, img2, img2))
