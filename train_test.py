import torch


def train(models, epochs, dataloader, optimizer, loss_fn):
    """
    Tranin the models

    Paramters
    ---------
    models: dictnoary
        The dictonary contains correspondence subnet and colorization subnet
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

    # Training
    for epoch in range(epochs):

        epoch_loss = 0

        for cut_i, cut in enumerate(dataloader):

            cut_loss = 0

            # Extract inputs from data
            frames = cut['L']
            gt = cut['Lab']
            ref = cut['ref']

            # Use GPU
            device = 'cuda'
            frames = frames.to(device)
            gt = gt.to(device)
            ref = ref.to(device)

            prev_ab = ref

            optimizer.zero_grad()

            # Iterate over frames in one cut
            for i in range(len(frames)):

                W_ab, S = models['cores'](frames[i], ref)

                pred_ab = models['color'](prev_ab, frames[i], W_ab, S)

                loss = loss_fn(pred_ab, prev_ab, gt, ref)

                # Accumulate loss for the whole cut
                cut_loss += loss

            # Cuts have varied length, therefore we take average
            cut_loss /= len(frames)

            cut_loss.backward()
            optimizer.step()

            epoch_loss += cut_loss.item()

        loss_history.append(epoch_loss)

    return loss_history


def test(models, frames, ref):
    """
    Tranin the models

    Paramters
    ---------
    models: dictnoary
        The dictonary contains correspondence subnet and colorization subnet
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

    for model in models.values():
        model.eval()

    # Use GPU
    device = 'cuda'
    frames = frames.to(device)
    ref = ref.to(device)

    prev_ab = ref

    preds_ab = []

    with torch.no_grad():
        # Iterate over frames in one cut
        for i in range(len(frames)):

            W_ab, S = models['cores'](frames[i], ref)

            pred_ab = models['color'](prev_ab, frames[i], W_ab, S)

            preds_ab.apppend(pred_ab)

    return preds_ab
