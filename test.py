from models.colornet import Colornet
from models.corresnet import CorrespodenceNet
from models.color import ExampleColorNet
from torch.utils.data import DataLoader
from skimage import color
import torch
import torchvision.transforms as transforms
import os
import cv2
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test Colorization Model')
parser.add_argument('--path', default='test', type=str, help='Path to testing data folder')
opt = parser.parse_args()

PATH = opt.path

def test(nets, frames, ref):
    """
    Tranin the models

    Paramters
    ---------
    nets: dictionary
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
    for name, net in zip(names, nets.values()):
        net.eval()
        if torch.cuda.is_available():
            net.cuda()
            net.load_state_dict(torch.load('checkpoints/' + name))
        else:
            net.load_state_dict(torch.load('checkpoints/' + name, map_location=lambda storage, loc: storage))

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

            W_ab, S = nets['corres'](frame, ref)
            pred = nets['color'](prev, frame, W_ab, S)

            pred = torch.cat((frame, pred), 1)
            prev = pred

            preds.append(pred[0])

    return preds

def load_image(path, size=(0, 0), mode='rgb'):
    '''
    Load an image from a path and process it if provided

    ----------
    Parameters:
    path: str
        Path to an image
    size: tuple of int
        New size of image
    mode: str ('gray, 'rgb', or 'lab')
        Color space of image

    ----------
    Return:
    An tensor of image with size [1, C, H, W]
    '''

    print(path)
    if mode == 'gray':
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if not size[0] == 0:
            img = cv2.resize(img, size)
        img = (img.astype(np.float32) - 127.) * 100./255. + 50.
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.imread(path)
        if not size[0] == 0:
            img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if mode == 'lab':
            img = color.rgb2lab(img).astype(np.float32)

    img = torch.from_numpy(img).permute(2, 0, 1)
    img[0] -= 50.

    return img.unsqueeze(0)

if __name__ == '__main__':
    # Prepare paths
    path = PATH
    paths_frame = glob.glob(path + '/frames/*')
    paths_frame.sort()
    path_ref = glob.glob(path + '/ref/*')

    # Prepare data
    for idx, path_frame in enumerate(paths_frame):
        img = load_image(path_frame, size=(112, 64), mode='gray')
        img_ori = load_image(path_frame, mode='gray')

        if idx == 0:
            frames = img
            ori = img_ori
        else:
            frames = torch.cat((frames, img), 0)
            ori = torch.cat((ori, img_ori), 0)

    ref = load_image(path_ref[0], size=(112, 64), mode='lab')
    
    # Prepare model
    #nets = {'corres': CorrespodenceNet(), 'color': Colornet()}
    nets = {'corres': CorrespodenceNet(), 'color': ExampleColorNet()}
    res = test(nets, frames, ref)
    print('Get result!')

    if not os.path.isdir(path + '/res'):
        os.mkdir(path + '/res')

    for idx, image in enumerate(res):
        if torch.cuda.is_available():
            image = image.cpu()
        image[0] += 50.
        image = image.permute(1, 2, 0).numpy().astype(np.float64)
        small = color.lab2rgb(image).astype(np.float32)
        small = (255*small).astype(np.uint8)
        small = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path + '/res/' + str(idx) + '.jpg', small)

        # Save upscale image
        big = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        big[:, :, 0] = ori[idx].numpy().astype(np.float64) + 50.  # Use the original L-channel
        big = color.lab2rgb(big).astype(np.float32)
        big = (255*big).astype(np.uint8)
        big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path + '/res/up_' + str(idx) + '.jpg', big)
    print('Done')