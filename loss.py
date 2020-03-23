import torch
import torch.nn as nn
import torchvision.models as models
import  torchvision.transforms as transforms

PERCEPTUAL_LAMDA = 0.001
CONTEXTUAL_LAMDA = 0.2
SMOOTHNESS_LAMDA = 5.0
FLOW_LAMBA = 0.02
L1_LAMBA = 0.1

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        # Extract vgg19 relu5_2 features
        vgg = models.vgg19_bn(pretrained=True)
        vgg_feature = nn.Sequential(*list(vgg.features)[:45]).eval()
        for param in vgg_feature.parameters():
            param.requires_grad = False
        self.vgg_feature = vgg_feature


    def forward(self, res, last_res, ground_truth, ref):
        '''
        Parameters:
        res: Tensor of image with size [3, H, W]
            Result frame in CIELAB color space from model .
        last_res: Tensor of image with size [3, H, W]
            Last result frame in CIELAB color space from model.
        ground_truth: Tensor of image with size [3, H, W]
            The ground truth in CIELAB color space of frame.
        ref: Tensor of image with size [3, H, W]
            Reference image in CIELAB color space of the cut.
        '''
        
        perceptual = self.perceptual_loss(res, ground_truth)
        contextual = self.contextual_loss(res, ref)
        smoothness = self.smoothness_loss(res, ground_truth)
        flow = self.flow_loss(res, last_res)
        l1 = self.l1_loss(res, ground_truth)
        
        return PERCEPTUAL_LAMDA*perceptual + CONTEXTUAL_LAMDA*contextual + SMOOTHNESS_LAMDA*smoothness + FLOW_LAMBA*flow + L1_LAMBA*l1


    def perceptual_loss(self, res, ground_truth):
        _res = res.unsqueeze(0)
        _ground_truth = ground_truth.unsqueeze(0)
        return (self.vgg_feature(_res) - self.vgg_feature(_ground_truth)).norm()


    def contextual_loss(self, res, ref):
        return 0


    def smoothness_loss(self, res, ground_truth):
        return 0


    def flow_loss(self, res, last_res):
        return 0
    
    
    def l1_loss(self, res, ground_truth):
        return 0


if __name__ == '__main__':
    import cv2

    # Load images
    img1 = cv2.imread('data/train/0/frames/0_0.jpg')
    img2 = cv2.imread('data/train/0/frames/0_1.jpg')
    # Resize to 480 x 272
    img1 = cv2.resize(img1, (480, 272))
    img2 = cv2.resize(img2, (480, 272))
    # Convert to CIELAB color space
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    
    # Convert to tensor with size [1, C, H, W] from [H, W, C]
    img1 = transforms.ToTensor()(img1)
    img2 = transforms.ToTensor()(img2)
    #print(img1.size())

    loss = Loss()
    print(loss(img1, img2, img2, img1))
