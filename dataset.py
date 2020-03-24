import torch
import cv2
import glob
import numpy as np
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, size=(64, 64), phase='train', transforms=None):
        '''
        Parameters:
        size: tuple of 2 unsigned integers
            New size of image.
        phase: str
            Phase to use data, can be either 'train' and 'test'.
        transforms: list of Transform object
            Transforms to be applied on a sample.
        '''

        super(Dataset, self).__init__()

        self.size = size
        self.phase = phase
        self.transforms = transforms
        self.folders = glob.glob('data/' + self.phase + '/*')
        self.folders.sort()


    def __len__(self):
        return len(self.folders)

    
    def __getitem__(self, index):
        '''
        Return:
        frames: dictionary, including:
            'L': Sequence of Tensors of image [N, 1, H, W]
                L-channel in CIELAB color space of frames in the cut.
            'Lab': Sequence of Tensors of image [N, 3, H, W]
                Lab-channel in CIELAB color space of frames in the cut.
            'ref': Tensor of image [3, H, W]
                Lab-channel in CIELAB color space of reference image of the cut.
        '''

        # Get random seed for all data
        seed = np.random.randint(774967893101)

        # Get paths to frames and reference image
        folder = self.folders[index]
        paths_frame = glob.glob(folder + '/frames/*')
        paths_frame.sort()
        path_ref = glob.glob(folder + '/ref/*')[0]

        for idx, path in enumerate(paths_frame):
            img = self.image(path, self.size, self.transforms, seed, color_mode=cv2.COLOR_BGR2LAB)

            if idx == 0:
                L = img[0].unsqueeze(0).unsqueeze(0)
                Lab = img.unsqueeze(0)
            else:
                L = torch.cat((L, img[0].unsqueeze(0).unsqueeze(0)), 0)
                Lab = torch.cat((Lab, img.unsqueeze(0)), 0)
        
        ref = self.image(path_ref, self.size, self.transforms, seed, color_mode=cv2.COLOR_BGR2LAB)

        return {'L': L, 'Lab': Lab, 'ref': ref} # Combine to a single dictionary


    def image(self, path, size, transforms, seed, color_mode=-1):
        '''
        Read image and process.

        ----------
        Parameters:
        path: str
            Path to image.
        size: tuple of 2 unsigned integers
            New size of the image.
        transforms:
            Transforms to be applied on a sample.
        seed: int
            Set the seed for random generator.
        color_mode: 
            Mode of converting color space in OpenCV2.

        ----------
        Return:
        Tensor of image with size [3, H, W]
        '''

        img = cv2.imread(path)
        img = cv2.resize(img, size)

        if not color_mode == -1:
            img = cv2.cvtColor(img, color_mode)
            
        if transforms:
            from PIL import Image

            img = Image.fromarray(img)
            random.seed(seed)
            img = transforms(img)
        else:
            img = torch.from_numpy(img).permute(1, 2, 0)

        return img


if __name__ == '__main__':
    import torchvision.transforms as transforms

    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor()
    ])

    data = Dataset(size=(480, 272), transforms=trans)
    frames = data[0]

    # Check size
    print(frames['L'].size())
    print(frames['Lab'].size())
    print(frames['ref'].size())

    img = frames['Lab'][0].permute(1, 2, 0)
    # Convert back to BGR image for visualizing
    img = 255*img.detach().numpy()
    img = img.astype(np.uint8)  # OpenCV supports uint8 for integer values
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # Visualize
    cv2.imshow('test', img)
    cv2.waitKey(0)