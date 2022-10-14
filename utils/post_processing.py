# ------------------------------------------------------------------------------
# Post-processing to get instance and panoptic segmentation results.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np

def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=None):
    """
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0)
    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1])

def our_instance(sem_seg, ctr_map, offsets, threshold, nms_kernel, grouping_dist):
    ctr = find_instance_center(ctr_map * sem_seg, threshold = threshold, nms_kernel = nms_kernel, top_k = None)
    _, height, width = sem_seg.shape
    if ctr.size(0) == 0: # no centers, no instances
        return torch.zeros_like(sem_seg)
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)
    coord = coord.reshape((2, height*width))
    bla = (coord.T.unsqueeze(0) - ctr.unsqueeze(1))
    bla2 = torch.sqrt(bla[:,:,0]**2 + bla[:,:,1]**2).reshape((ctr.shape[0], height,width))
    dist = torch.sqrt( offsets[0][0]**2 + offsets[0][1]**2)
    single_masks = torch.isclose(bla2, dist, atol = grouping_dist)
    non_overlapping = torch.zeros((height,width)).cuda()
    for mask in single_masks:
        non_overlapping += mask
    non_overlapping[ non_overlapping != 1] = 0

    ins_seg_test = torch.zeros((height, width)).cuda()
    for i, mask in enumerate(single_masks):
        ins_seg_test += mask*(i+1)

    pixels = torch.where( torch.logical_not(non_overlapping)*sem_seg[0] )
    if len(pixels[0]) != 0 and (ins_seg_test * sem_seg[0] * non_overlapping).sum() != 0: # there are overlapping pixels
        pixels = torch.as_tensor( torch.cat((pixels[0].unsqueeze(1) , pixels[1].unsqueeze(1)), dim=1))

        label = torch.where( ins_seg_test*sem_seg[0]*non_overlapping != 0 )
        label = torch.as_tensor(torch.cat((label[0].unsqueeze(1) , label[1].unsqueeze(1)), dim=1))
        
        try:
            d = torch.cdist(pixels.unsqueeze(0).float(), label.unsqueeze(0).float())
            d = torch.argsort(d.squeeze(0))[:,:5] 
        except:
            d = torch.cdist(pixels.unsqueeze(0).float().cpu(), label.unsqueeze(0).float().cpu())
            d = torch.from_numpy( np.argsort(d.squeeze(0).cpu().numpy())[:,:5]).cuda() 

        value = ins_seg_test[ label[d,0] , label[d,1] ].long()
        N = value.max() + 1
        offs = value + torch.arange(value.shape[0])[:,None].cuda() * N
        count = torch.bincount(offs.ravel(), minlength=value.shape[0]*N).reshape(-1,N)
        count = torch.argmax(count, axis=1)
        ins_seg_test[ pixels[:,0], pixels[:,1] ] = count.float()
    
    return ins_seg_test*sem_seg[0]
