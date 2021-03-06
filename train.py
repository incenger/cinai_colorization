from dataset import Dataset
from torch.utils.data import DataLoader
from models.colornet import ColorizationNet
from models.corresnet import CorrespodenceNet
from models.color import AdditionalColorNet
from loss import Loss
from torch.optim import Adam
import torch
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='Train Colorization Model')
parser.add_argument('-p', '--path', default='data/train', type=str, help='Path to the training data folder')
parser.add_argument('-n', '--num_epochs', default=20, type=int, help='Number of epochs to train')
opt = parser.parse_args()

PATH = opt.path
EPOCHS = opt.num_epochs
LEARNING_RATE = 2*1e-4
BETAS = (0.5, 0.999)

def train(nets, epochs, dataloader, optimizer, loss_fn):
    """
    Train the models

    Paramters
    ---------
    nets: dictionary
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

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    loss_history = []

    # Setting models to train mode

    for net in nets.values():
        net.train()
        if torch.cuda.is_available():
            net.cuda()

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

                W_ab, S = nets['corres'](frame, ref)
                pred = nets['color'](prev, frame, W_ab, S)

                pred = torch.cat((frame, pred), 1)
                prev = gt if epoch < 5 else pred

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

        # Save weights
        torch.save(nets['corres'].state_dict(), 'checkpoints/corresnet.pth')
        torch.save(nets['color'].state_dict(), 'checkpoints/colornet.pth')

    return loss_history

if __name__ == '__main__':
    # Prepare data
    data = Dataset(path=PATH, size=(112, 64))
    cutloader = DataLoader(data, batch_size=1, num_workers=4, shuffle=True)

    # Prepare model, optim, loss
    #nets = {'corres': CorrespodenceNet(), 'color': ColorizationNet()}
    nets = {'corres': CorrespodenceNet(), 'color': AdditionalColorNet()}
    params = list(nets['corres'].parameters()) + list(nets['color'].parameters())
    optimizer = Adam(params, lr=LEARNING_RATE, betas=BETAS, amsgrad=True)
    loss_fn = Loss()

    with torch.autograd.set_detect_anomaly(True):
        history = train(nets, EPOCHS, cutloader, optimizer, loss_fn)

    if not os.path.isdir('result'):
        os.mkdir('result')
    # Show loss
    plt.plot(history, label='train')
    plt.title('Loss')
    plt.legend()
    plt.savefig('result/loss.jpg')
    plt.show()
