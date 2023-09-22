import torch
import torch.nn as nn

class Dice_Segmentation_Loss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(Dice_Segmentation_Loss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # predictions to probabilities
        probs = torch.sigmoid(preds)
        b_size = probs.shape[0]

        probs = probs.view(b_size, -1)
        targets = targets.view(b_size, -1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coefficient

        return dice_loss
