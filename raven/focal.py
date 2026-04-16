from torchvision.ops import sigmoid_focal_loss
import torch.nn as nn
from torchvision.ops import focal_loss


class FocalLoss(nn.Module):

    def __init__(self, gamma=2,alpha=0.75, reduction = "none"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
    
        return sigmoid_focal_loss(inputs = inputs, targets = targets , alpha = self.alpha, gamma = self.gamma, reduction = self.reduction)