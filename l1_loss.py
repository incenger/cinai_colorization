import torch
import torch.nn as nn

class L1Loss(nn.module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss_fn = nn.L1Loss(reduction='mean')

    def forward(self, pred_ab, gt_ab):
        return self.loss_fn(pred_ab, gt_ab)
