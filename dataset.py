import torch
import torchvision.transforms as transforms
import cv2
import glob
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase='train', transforms=None):
        '''
        Parameters:
        phase: str
            Phase to use data, can be either 'train' and 'test'.
        transforms: list of Transform object
            Transforms to be applied on a sample.
        '''

        super(Dataset, self).__init__()

        self.phase = phase
        self.transforms = transforms


    def __len__(self):
        return len(glob.glob('data/' + phase + '/*'))

    
    def __getitem__(self, index):
        '''
        Return:
        frames: Tensor of images with size [N, C, H, W]
            Sequence of frames in the cut
        ref: Tensor of image with siez [C, H, W]
            Reference image of the cut
        '''

        path = 'data/' + self.phase + '/' + str(index) + '/'
        num_frames = len(glob.glob(path + 'frames/*'))

        frames = []
        for i in range(num_frames):
            img = cv2.imread(path + 'frames/' + str(index) + '_' + str(i) +'.jpg')
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        
        ref = cv2.imread(path + 'ref/ref_' + str(index) + '.jpg')
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)

        # Transform data to tensor
        frames = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2) # Convert [N, H, W, C] to [N, C, H, W]
        ref = transforms.ToTensor()(ref)                                # Convert to tensor with size [C, H, W]
        
        return frames, ref

if __name__ == '__main__':
    data = Dataset()
    frames, ref = data[0]

    # Check size
    print(frames.size())
    print(ref.size())

    # Visualize
    img = frames[0].data.numpy()
    img = img.transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    cv2.imshow('test', img)
    cv2.waitKey(0)