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
        res: 
            Result frame from model 
        last_res: 
            Last result frame from model
        ground_truth:
            The ground truth of frame
        ref:
            Reference image of the cut
        '''
        
        perceptual = self.perceptual_loss(res, ground_truth)
        contextual = self.contextual_loss(res, ref)
        smoothness = self.smoothness_loss(res, ground_truth)
        flow = self.flow_loss(res, last_res)
        l1 = self.l1_loss(res, ground_truth)
        
        return PERCEPTUAL_LAMDA*perceptual + CONTEXTUAL_LAMDA*contextual + SMOOTHNESS_LAMDA*smoothness + FLOW_LAMBA*flow + L1_LAMBA*l1


    def perceptual_loss(self, res, ground_truth):
        return (self.vgg_feature(res) - self.vgg_feature(ground_truth)).norm()


    def contextual_loss(self, res, ref):
        return 0


    def smoothness_loss(self, res, ground_truth):
        return 0


    def flow_loss(self, res, last_res):
        return 0
    
    
    def l1_loss(self, res, ground_truth):
        loss_fn = nn.L1Loss(reduction="sum")
        return loss_fn(res, ground_truth)


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
