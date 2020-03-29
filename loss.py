from skimage import color
import torch
import torch.nn as nn
import torchvision.models as models

PERCEPTUAL_LAMDA = 0.001
CONTEXTUAL_LAMDA = 0.2
SMOOTHNESS_LAMDA = 5.0
FLOW_LAMBA = 0.02
L1_LAMBA = 2.0

EPSILON = 1e-5
BANDWITH = 0.1

class Loss(nn.Module):
    def __init__(self, requires_grad=False):
        super(Loss, self).__init__()

        vgg = models.vgg19_bn(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = False

        # relu2_2: 12, relu3_2: 19, relu4_2: 32, reku5_2: 45
        self.vgg19_relu2_2 = nn.Sequential(vgg.features[:13]).eval()
        self.vgg19_relu3_2 = nn.Sequential(vgg.features[:20]).eval()
        self.vgg19_relu4_2 = nn.Sequential(vgg.features[:33]).eval()
        self.vgg19_relu5_2 = nn.Sequential(vgg.features[:46]).eval()

        self.w = [0.1, 0.6, 0.7, 0.8]
        self.N = [128, 256, 512, 512]


    def forward(self, res, last_res, gt, ref):
        '''
        Parameters:
        res: Tensor of image with size [1, 3, H, W]
            Result frame in CIELAB color space from model .
        last_res: Tensor of image with size [1, 3, H, W]
            Last result frame in CIELAB color space from model.
        gt: Tensor of image with size [1, 3, H, W]
            The ground truth in CIELAB color space of frame.
        ref: Tensor of image with size [1, 3, H, W]
            Reference image in CIELAB color space of the cut.
        '''
        
        perceptual = self.perceptual_loss(res, gt)
        contextual = self.contextual_loss(res, ref)
        smoothness = self.smoothness_loss(res, gt)
        flow = self.flow_loss(res, last_res)
        l1 = self.l1_loss(res, gt)
        
        return PERCEPTUAL_LAMDA*perceptual + CONTEXTUAL_LAMDA*contextual + SMOOTHNESS_LAMDA*smoothness + FLOW_LAMBA*flow + L1_LAMBA*l1


    def perceptual_loss(self, res, gt):
        return (self.vgg19_relu5_2(res) - self.vgg19_relu5_2(gt)).norm()**2


    def contextual_loss(self, res, ref):
        context = 0
        for idx, feature in enumerate([self.vgg19_relu2_2, self.vgg19_relu3_2, self.vgg19_relu4_2, self.vgg19_relu5_2]):
            x_hat = feature(res).reshape((self.N[idx], -1)).T
            x_hat = (x_hat - x_hat.mean(dim=0, keepdim=True)) / (x_hat.norm(dim=0, keepdim=True) + EPSILON)
            y = feature(ref).reshape((self.N[idx], -1)).T
            y = (y - y.mean(dim=0, keepdim=True)) / (y.norm(dim=0, keepdim=True) + EPSILON)
            log_inside = 0
            for row_i in x_hat:
                cosine_dist = torch.mm(row_i.unsqueeze(0), y.T)
                cosine_dist = cosine_dist / (cosine_dist.min() + EPSILON)
                affinity = (nn.Softmax(dim=1)((1-cosine_dist) / BANDWITH)).max()
                log_inside += affinity
            context += self.w[idx] * (-((1./self.N[idx] * log_inside).log()))
        return context

    def smoothness_loss(self, res, gt):
        return 0


    def flow_loss(self, res, last_res):
        return 0
    
    
    def l1_loss(self, res, gt):
        _res = res[:, 1:]
        _gt = gt[:, 1:]
        loss_fn = nn.L1Loss(reduction="sum")
        return loss_fn(_res, _gt)


if __name__ == '__main__':
    import cv2
    import numpy as np
    import torchvision.transforms as transforms
    from skimage import color

    # Load images
    img1 = cv2.imread('data/cut_000/frames/000_750.jpg')
    img2 = cv2.imread('data/cut_000/frames/000_738.jpg')
    #img2 = cv2.imread('data/cut_005/frames/005_8544.jpg')
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


    loss = Loss()
    print(loss(img1, img2, img2, img1))
