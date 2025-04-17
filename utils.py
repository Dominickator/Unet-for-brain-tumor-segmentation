import torch

def dice_score(pred, target, threshold=0.5):
    """
    Computes Dice score for binary segmentation.

    Args:
        pred: predicted mask (BCHW or CHW)
        target: ground truth mask
        threshold: threshold to binarize prediction

    Returns:
        Dice coefficient (float)
    """
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
