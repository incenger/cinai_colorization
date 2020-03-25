from models.colornet import Colornet
from models.corresnet import CorrespodenceNet
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import os
import cv2
import glob
import numpy as np

def test(models, frames, ref):
    """
    Tranin the models

    Paramters
    ---------
    models: dictionary
        The dictionary contains correspondence subnet and colorization subnet
    frames: tensor
        Frames to color
    ref : tensor
        Reference image

    Returns
    -------
    preds_ab: Tensor
        List of predicted ab channel corresponding to the frame
    """

    # Setting models to evaluate mode
    names = ['corresnet.pth', 'colornet.pth']
    for name, model in zip(names, models.values()):
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
            model.load_state_dict(torch.load('checkpoints/' + name))
        else:
            model.load_state_dict(torch.load('checkpoints/' + name, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        ref = ref.cuda()

    frameloader = DataLoader(frames, batch_size=1)

    # Initialize previous frame with reference image
    prev = ref

    preds = []

    with torch.no_grad():
        # Iterate over frames in one cut
        for frame in frameloader:
            if torch.cuda.is_available():
                frame = frame.cuda()

            W_ab, S = models['corres'](frame, ref)

            pred = models['color'](ref[:, 1:], frame, W_ab, S)
            prev = pred
            
            preds.append(pred[0])

    return preds

def load_image(path, size=(64, 64), color_mode=-1):
    print(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, size)

    if not color_mode == -1:
        img = cv2.cvtColor(img, color_mode)

    img = transforms.ToTensor()(img)

    return img.unsqueeze(0)

if __name__ == '__main__':
    # Prepare paths
    path = 'test' # Path to test folder
    paths_frame = glob.glob(path + '/frames/*')
    paths_frame.sort()
    path_ref = glob.glob(path + '/ref/*')

    # Prepare data
    for idx, path_frame in enumerate(paths_frame):
        img = load_image(path_frame)

        if idx == 0:
            frames = img
        else:
            frames = torch.cat((frames, img), 0)

    ref = load_image(path_ref[0], color_mode=cv2.COLOR_BGR2LAB)
    
    # Prepare model
    models = {'corres': CorrespodenceNet(), 'color': Colornet()}
    res = test(models, frames, ref)

    if not os.path.isdir(path + '/res'):
        os.mkdir(path + '/res')

    for idx, image in enumerate(res):
        if torch.cuda.is_available():
            image = image.cpu()
        image = 255*image.permute(1, 2, 0).numpy()  # ToTensor() normalize image, convert it back
        image = image.astype(np.uint8)              # OpenCV2 supports uint8
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        cv2.imwrite(path + '/res/' + str(idx) + '.jpg', image)

    print('Done')