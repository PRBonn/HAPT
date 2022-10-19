from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F 
import cv2 as cv
import scipy.ndimage as spni
import numpy as np
import random 

""" CrossEntropy loss, usually used for semantic segmentation.
"""

class CrossEntropyLoss(nn.Module):

  def __init__(self, weights: Optional[List] = None):
    super().__init__()
    if weights is not None:
      weights = torch.Tensor(weights)
    self.criterion = nn.CrossEntropyLoss(weights)

  def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ Compute cross entropy loss.

    Args:
        inputs (torch.Tensor): unnormalized input tensor of shape [B x C x H x W]
        target (torch.Tensor): ground-truth target tensor of shape [B x H x W]

    Returns:
          torch.Tensor: weighted mean of the output losses.
    """

    loss = self.criterion(inputs, target)

    return loss

""" Generalized IoU loss.
"""

class mIoULoss(nn.Module):
  """ Define mean IoU loss.

  Props go to https://github.com/PRBonn/bonnetal/blob/master/train/tasks/segmentation/modules/custom_losses.py
  """
  def __init__(self, weight: List[float]):
    super().__init__()
    self.weight = nn.Parameter(torch.Tensor(weight), requires_grad=False)

  def forward(self, logits: torch.Tensor, target: torch.Tensor, masking: torch.Tensor) -> torch.Tensor:
    """ Compute loss based on predictions/inputs and ground-truths/targets.

    Args:
        logits (torch.Tensor): Predictions of shape [N x n_classes x H x W]
        target (torch.Tensor): Ground-truths of shape [N x H x W]

    Returns:
        torch.Tensor: mean IoU loss
    """
    # get number of classes
    n_classes = int(logits.shape[1])

    # target to onehot
    target_one_hot = self.to_one_hot(target, n_classes) # [N x H x W x n_classes]

    batch_size = target_one_hot.shape[0]

    # map to (0,1)
    probs = F.softmax(logits, dim=1)
    
    # Numerator Product
    inter = probs * target_one_hot * masking.unsqueeze(1)

    # Average over all pixels N x C x H x W => N x C
    inter = inter.view(batch_size, n_classes, -1).mean(2) + 1e-8

    # Denominator
    union = probs + target_one_hot - (probs * target_one_hot) + 1e-8
    # Average over all pixels N x C x H x W => N x C
    union = union.view(batch_size, n_classes, -1).mean(2)

    # Weights for loss
    frequency = target_one_hot.view(batch_size, n_classes, -1).sum(2).float()
    present = (frequency > 0).float()

    # -log(iou) is a good surrogate for loss
    loss = -torch.log(inter / union) * present * self.weight
    loss = loss.sum(1) / present.sum(1)  # pseudo average

    # Return average loss over batch
    return loss.mean()

  def to_one_hot(self, tensor: torch.Tensor, n_classes: int) -> torch.Tensor:
    """ Convert tensor to its one hot encoded version.

    Props go to https://github.com/PRBonn/bonnetal/blob/master/train/common/onehot.py

    Args:
      tensor (torch.Tensor): ground truth tensor of shape [N x H x W]
      n_classes (int): number of classes

    Returns:
      torch.Tensor: one hot tensor of shape [N x n_classes x H x W]
    """
    if len(tensor.size()) == 1:
        b = tensor.size(0)
        if tensor.is_cuda:
            one_hot = torch.zeros(b, n_classes, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(b, n_classes).scatter_(1, tensor.unsqueeze(1), 1)
    elif len(tensor.size()) == 2:
        n, b = tensor.size()
        if tensor.is_cuda:
            one_hot = torch.zeros(n, n_classes, b, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(n, n_classes, b).scatter_(1, tensor.unsqueeze(1), 1)
    elif len(tensor.size()) == 3:
        n, h, w = tensor.size()
        if tensor.is_cuda:
            one_hot = torch.zeros(n, n_classes, h, w, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.unsqueeze(1), 1)
    return one_hot
                                                                                                                          
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=.01, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, out, target, masking):
        out = out.squeeze() * masking.squeeze()
        target = target.squeeze()
        log_positives = (out[target != 0]).clamp(min=1e-4)
        log_negatives = (1 - out[target == 0]).clamp(min=1e-4)
        positives = -self.alpha * (1 - out[target != 0]) ** self.gamma * torch.log(log_positives)
        negatives = -(1 - self.alpha) * out[target == 0] ** self.gamma * torch.log(log_negatives)
        if len(positives) > 0 and len(negatives) > 0:
            return torch.mean(positives) + torch.mean(negatives)
        elif len(positives) > 0:
            return torch.mean(positives)
        return torch.mean(negatives)

def masks_to_centers(masks_original: torch.Tensor) -> torch.Tensor:
    if masks_original.numel() == 0:
        return torch.zeros((0, 4), device=masks_original.device, dtype=torch.float)

    tmp_masks = F.one_hot(masks_original.long()).permute(0,3,1,2)
    masks = tmp_masks[:,1:,:,:]
    B, num, H, W = masks.shape

    center_mask = torch.zeros( (B, H, W) , device=masks.device, dtype=torch.float)
    
    for batch_idx, mask in enumerate(masks):
        for submask in mask:
            if submask.sum() == 0:
                continue
            x, y = torch.where(submask != 0)
            xy = torch.cat([x.unsqueeze(0),y.unsqueeze(0)], dim=0)
            mu, _ = torch.median(xy,dim=1, keepdim=True)
            center_idx = torch.argmin(torch.sum(torch.abs(xy - mu), dim=0))
            center = xy[:,center_idx]
            center_mask[batch_idx, center[0], center[1]] = 1.
    return center_mask
