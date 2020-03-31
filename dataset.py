from skimage import color
import torch
import torchvision.transforms as transforms
import cv2
import glob
import numpy as np
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, size=(64, 64)):
        '''
        Parameters:
        size: tuple of 2 unsigned integers
            New size of image.
        path: str
            Path to data folder.
        '''

        super(Dataset, self).__init__()

        self.size = size
        self.folders = glob.glob(path + '/*')
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
            'ref': Tensor of image [1, 3, H, W]
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
            img = self.image(path, self.size, seed)

            if idx == 0:
                L = img[:, 0].unsqueeze(0)
                Lab = img
            else:
                L = torch.cat((L, img[:, 0].unsqueeze(0)), 0)
                Lab = torch.cat((Lab, img), 0)
        
        ref = self.image(path_ref, self.size, seed)

        return {'L': L, 'Lab': Lab, 'ref': ref} # Combine to a single dictionary


    def image(self, path, size, seed):
        '''
        Read image and process.

        ----------
        Parameters:
        path: str
            Path to image.
        size: tuple of 2 unsigned integers
            New size of the image.
        seed: int
            Set the seed for random generator.

        ----------
        Return:
        Tensor of image with size [1, 3, H, W]
        '''

        # Load image, resize and convert to CIELAB
        img = cv2.imread(path)
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = color.rgb2lab(img).astype(np.float32)
            
        random.seed(seed)
        # Random horizontal and vertical
        # Not yet implement
        img = torch.from_numpy(img).permute(2, 0, 1)    # Convert [H, W, C] to [C, H, W]
        img[0] -= 50.   # Subtract mean values for VGG19

        return img.unsqueeze(0)


if __name__ == '__main__':
    data = Dataset('data', size=(480, 272))
    frames = data[0]

    # Check size
    print(frames['L'].size())
    print(frames['Lab'].size())
    print(frames['ref'].size())

    # Convert back to RGB
    img = frames['Lab'][0].detach()
    img[0] += 50.
    img = img.permute(1, 2, 0).numpy()
    img = color.lab2rgb(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('abc', img)
    cv2.waitKey(0)