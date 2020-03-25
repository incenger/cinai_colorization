from models.colornet import Colornet
from models.corresnet import CorrespodenceNet
import torch

def test(nets, frames, ref):
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

    for model in nets.values():
        model.eval()
        if torch.cuda.is_available():
            model.cuda()

    if torch.cuda.is_available():
        frames.cuda()
        ref.cuda()

    prev_ab = ref

    preds_ab = []

    with torch.no_grad():
        # Iterate over frames in one cut
        for i in range(len(frames)):

            W_ab, S = nets['cores'](frames[i], ref)

            pred_ab = nets['color'](prev_ab, frames[i], W_ab, S)

            preds_ab.apppend(pred_ab)

    return preds_ab
