from dataset import Dataset
from torch.utils.data import DataLoader
from models.colornet import Colornet
from models.corresnet import CorrespodenceNet
from loss import Loss
from torch.optim import Adam
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import os

EPOCHS = 10

def train(models, epochs, dataloader, optimizer, loss_fn):
    """
    Tranin the models

    Paramters
    ---------
    models: dictionary
        The dictionary contains correspondence subnet and colorization subnet
    epochs: int
        Number of epochs to train
    dataloader:
        The dataloader
    optimizer:
        The optimizer to run
    loss_fn:
        The overall loss module

    Returns
    -------
    loss_history : list
        The loss of each epoch
    """

    loss_history = []

    # Setting models to train mode

    for model in models.values():
        model.train()
        if torch.cuda.is_available():
            model.cuda()

    if torch.cuda.is_available():
        loss_fn.cuda()

    # Training
    for epoch in range(epochs):
        print('Epoch ', epoch)

        epoch_loss = 0

        for cut_i, cut in enumerate(dataloader):

            cut_loss = 0

            # Extract inputs from data
            frames = cut['L'][0] # [N, 1, H, W]
            gt = cut['Lab'][0]   # [N, 3, H, W]
            ref = cut['ref'][0]  # [1, 3, H, W]

            if torch.cuda.is_available():
                ref = ref.cuda()

            frameloader = DataLoader(frames, batch_size=1)
            gtloader = DataLoader(gt, batch_size=1)

            # Initialize previous frame with reference image
            prev = ref

            optimizer.zero_grad()

            # Iterate over frames in one cut
            for frame, gt in zip(frameloader, gtloader):
                if torch.cuda.is_available():
                    frame = frame.cuda()
                    gt = gt.cuda()

                W_ab, S = models['corres'](frame, ref)

                pred = models['color'](ref[:, 1:], frame, W_ab, S)
                prev = pred

                loss = loss_fn(pred, prev, gt, ref)

                # Accumulate loss for the whole cut
                cut_loss += loss

            # Cuts have varied length, therefore we take average
            cut_loss /= len(frames)

            print('Cut ', cut_i, ' Loss ', cut_loss.item())

            cut_loss.backward()
            optimizer.step()

            epoch_loss += cut_loss.item()

        print('Epoch loss ', epoch_loss)

        loss_history.append(epoch_loss)

    # Save model
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(models['corres'].state_dict(), 'checkpoints/corresnet.pth')
    torch.save(models['color'].state_dict(), 'checkpoints/colornet.pth')

    return loss_history

if __name__ == '__main__':
    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    # Prepare data
    data = Dataset(path='data', transforms=trans)
    cutloader = DataLoader(data, batch_size=1, num_workers=4, shuffle=True)

    # Prepare model, optim, loss
    models = {'corres': CorrespodenceNet(), 'color': Colornet()}
    params = list(models['corres'].parameters()) + list(models['color'].parameters())
    optimizer = Adam(params)
    loss_fn = Loss()

    with torch.autograd.set_detect_anomaly(True):
        history = train(models, EPOCHS, cutloader, optimizer, loss_fn)

    if not os.path.isdir('result'):
        os.mkdir('result')
    # Show loss
    plt.plot(history, label='train')
    plt.title('Loss')
    plt.legend()
    plt.savefig('result/loss.jpg')
    plt.show()
