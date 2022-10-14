'''
Hierarchical Agriculture Perception Tasks
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from models.resnet import ResNet18, ResNet34, NonBottleneck1D
from pytorch_lightning.core.lightning import LightningModule
from models.blocks import DownsamplerBlock, UpsamplerBlock, non_bottleneck_1d
from models.loss import mIoULoss, BinaryFocalLoss
from torchmetrics import IoU
from utils.panoptic_quality import PanopticQuality as pq
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.nn.functional import one_hot
from utils.post_processing import our_instance
import matplotlib.pyplot as plt

class Hapt(LightningModule):

    def __init__(self, cfg: dict):
        super().__init__()
        self.n_classes = cfg['tasks']['semantic_segmentation']['n_classes']
        self.dropout = cfg['model']['dropout']
        self.epochs = cfg['train']['max_epoch']
        self.warmup = cfg['train']['validation_warmup']

        self.lr = cfg['train']['lr']
        self.init = cfg['model']['initialization']

        self.sem_loss = mIoULoss([1, 10])
        self.iou = IoU(num_classes=2, reduction='none')
        self.accumulated_miou = torch.tensor([0.0, 0.0]).cuda()

        self.plant_center_loss = BinaryFocalLoss()
        self.plant_offset_loss = nn.L1Loss(reduction="none") 
        self.leaf_center_loss =  BinaryFocalLoss()
        self.leaf_offset_loss = nn.L1Loss(reduction="none") 
        
        self.accumulated_iou_loss = 0.0
        self.accumulated_total_loss = 0.0
        self.accumulated_plant_offset_loss = 0.0
        self.accumulated_leaf_offset_loss = 0.0
        self.accumulated_plant_cen_loss = 0.0
        self.accumulated_leaf_cen_loss = 0.0
        self.val_loss = 0.0

        self.accumulated_pq_plants = 0.0
        self.accumulated_pq_leaves = 0.0

        self.pq_p = pq()
        self.pq_l = pq()

        # encoder
        self.encoder = ERFNetEncoder(self.n_classes, dropout=self.dropout, init=self.init)

        # decoders
        self.decoder_semseg = DecoderSemanticSegmentation(self.n_classes, self.dropout, init=self.init)
        self.decoder_plant = DecoderInstance(self.dropout, init=self.init)
        self.decoder_leaf = DecoderInstance(self.dropout, init=self.init)


    def forward(self, input):
        # encoder -- possibly from pre-trained weights
        out = self.encoder(input)
        
        #encoder_lr = self.scheduler_encoder.get_last_lr()
        #self.logger.experiment.add_scalar("LR/encoder", encoder_lr[0], self.trainer.current_epoch)

        # semantic segmentation
        semantic, skips = self.decoder_semseg(out)
        
        plant_centers, plant_offsets, skips = self.decoder_plant((out[-1], skips[0], skips[1]))
        
        leaf_centers, leaf_offsets, _ = self.decoder_leaf((out[-1], skips[0], skips[1]))
        
        return semantic, plant_centers, plant_offsets, leaf_centers, leaf_offsets

    def getLoss(self, sem, plant_centers, plant_offsets, leaf_centers, leaf_offsets, p_labels, p_centers, p_offsets, l_centers, l_offsets, loss_masking=None, is_train=True):
        gt = p_labels.bool().long()
        
        sem_loss = self.sem_loss(sem, gt, loss_masking)
        plant_cen_loss = self.plant_center_loss(plant_centers.squeeze(), p_centers.squeeze(), loss_masking)
        plant_offset_loss = (self.plant_offset_loss(plant_offsets * loss_masking.unsqueeze(1), p_offsets) * gt.unsqueeze(1)).mean()
        
        leaf_cen_loss = self.leaf_center_loss(leaf_centers.squeeze(), l_centers.squeeze(), loss_masking)
        leaf_offset_loss = (self.leaf_offset_loss(leaf_offsets * loss_masking.unsqueeze(1), l_offsets) * gt.unsqueeze(1)).mean() 

        total_loss = sem_loss + 10 * plant_cen_loss + 10 * plant_offset_loss + 75 * leaf_cen_loss + 10 * leaf_offset_loss

        if is_train:
            self.accumulated_iou_loss += sem_loss.detach()
            self.accumulated_plant_offset_loss += plant_offset_loss.detach()
            self.accumulated_leaf_offset_loss += leaf_offset_loss.detach()
            self.accumulated_plant_cen_loss += plant_cen_loss.detach()
            self.accumulated_leaf_cen_loss += leaf_cen_loss.detach()
            self.accumulated_total_loss += total_loss.detach()
        else:
            self.val_loss += total_loss.detach()

        return total_loss

    def training_epoch_end(self, training_step_outputs):

        n_samples = float(len(self.train_dataloader()))

        sem_loss = self.accumulated_iou_loss / n_samples
        self.accumulated_plant_offset_loss /= n_samples
        self.accumulated_leaf_offset_loss /= n_samples
        self.accumulated_plant_cen_loss /= n_samples
        self.accumulated_leaf_cen_loss /= n_samples

        total_loss = self.accumulated_total_loss / n_samples

        # tensorboard logs
        self.logger.experiment.add_scalar(
            "Loss/sem_loss", sem_loss, self.trainer.current_epoch)
        self.logger.experiment.add_scalar(
            "Loss/plant_cen_loss", self.accumulated_plant_cen_loss, self.trainer.current_epoch)
        self.logger.experiment.add_scalar(
            "Loss/leaf_cen_loss", self.accumulated_leaf_cen_loss, self.trainer.current_epoch)
        self.logger.experiment.add_scalar(
            "Loss/plant_offset_loss", self.accumulated_plant_offset_loss, self.trainer.current_epoch)
        self.logger.experiment.add_scalar(
            "Loss/leaf_offset_loss", self.accumulated_leaf_offset_loss, self.trainer.current_epoch) 
        self.logger.experiment.add_scalar(
            "Loss/total_loss", total_loss, self.trainer.current_epoch)

        self.accumulated_iou_loss *= 0.0
        self.accumulated_total_loss *= 0.0
        self.accumulated_plant_cen_loss *= 0.0
        self.accumulated_leaf_cen_loss *= 0.0
        self.accumulated_plant_offset_loss *= 0.0
        self.accumulated_leaf_offset_loss *= 0.0
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch['image'].float()
        sem, plant_centers, plant_offsets, leaf_centers, leaf_offsets = self.forward(x)
        loss = self.getLoss(sem, plant_centers, plant_offsets, leaf_centers, leaf_offsets, batch['global_instances'],
                            batch['global_centers'], batch['global_offsets'],  batch['parts_centers'], batch['parts_offsets'], batch['loss_masking'])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image'].float()
        p_labels = batch['global_instances']
        l_labels = batch['parts_instances']

        sem, plant_centers, plant_offsets, leaf_centers, leaf_offsets = self.forward(x)
        gt_batch = p_labels.bool().long()
        iou = self.iou(sem, gt_batch)

        _ = self.getLoss(sem, plant_centers, plant_offsets, leaf_centers, leaf_offsets, batch['global_instances'],
                         batch['global_centers'], batch['global_offsets'],  batch['parts_centers'], batch['parts_offsets'], batch['loss_masking'], False)

        # we sum all the ious, because in validation_epoch_end we divide by the number of samples = mean over validation set
        self.accumulated_miou += iou

        if True: #self.trainer.current_epoch >= self.warmup or (self.trainer.current_epoch % 30 == 0 and self.trainer.current_epoch > 70):
            for item in range(sem.shape[0]):
                semantic = torch.argmax(torch.softmax(sem[item],0),0)
                max_p_cen = torch.max(plant_centers[item])
                max_l_cen = torch.max(leaf_centers[item])
                plant_instance = our_instance(semantic.unsqueeze(0), plant_centers[item].unsqueeze(0), plant_offsets[item].unsqueeze(0), threshold=0.85*max_p_cen, nms_kernel=41, grouping_dist = 20.0)
                # plant panoptic quality
                self.pq_p.reset()
                p_pq_v, _ = self.pq_p.compute_pq(semantic, gt_batch[item], plant_instance.squeeze(), p_labels[item])
                            
                leaf_instance = our_instance(semantic.unsqueeze(0), leaf_centers[item].unsqueeze(0), leaf_offsets[item].unsqueeze(0), threshold=0.75*max_l_cen, nms_kernel=11, grouping_dist=2.)

                self.pq_l.reset()
                l_pq_v, _ = self.pq_l.compute_pq(semantic, gt_batch[item], leaf_instance.squeeze(), l_labels[item])
                if np.isnan(l_pq_v):
                        import ipdb; ipdb.set_trace()
 
                self.accumulated_pq_plants += p_pq_v
                self.accumulated_pq_leaves += l_pq_v

                # if batch_idx in [0, 21, 89, 318]:
                #     self.logger.experiment.add_image("Sem_mask/" + "b" + str(batch_idx) + "s" + str(item), semantic.unsqueeze(0), self.trainer.current_epoch)
                #     colormap = cm.get_cmap('tab20b')
                #     colormap.colorbar_extend = True

                #     ## these two lines are only if erfnet base encoder 
                #     plant_instance = plant_instance.squeeze()
                #     leaf_instance = leaf_instance.squeeze()
                    
                #     Ip_transformed = colormap(plant_instance.long().cpu())[:, :, :3]
                #     Il_transformed = colormap(leaf_instance.long().cpu())[:, :, :3]
                #     self.logger.experiment.add_image("Plant_mask/" + "b" + str(batch_idx) + "s" + str(item), Ip_transformed, self.trainer.current_epoch, dataformats='HWC')
                #     self.logger.experiment.add_image("Leaf_mask/" + "b" + str(batch_idx) + "s" + str(item), Il_transformed, self.trainer.current_epoch, dataformats='HWC')

            
    def validation_epoch_end(self, validation_step_outputs):
        n_batches = len(self.val_dataloader())
        n_samples = float(len(self.val_dataloader().dataset))

        self.accumulated_miou /= n_batches
        self.val_loss /= n_samples
        self.accumulated_pq_plants /= n_samples
        self.accumulated_pq_leaves /= n_samples

        self.logger.experiment.add_scalars(
            "Metrics/iou", {'soil': self.accumulated_miou[0], 'plants': self.accumulated_miou[1]}, self.trainer.current_epoch)
        self.logger.experiment.add_scalar(
            "Loss/val_total_loss", self.val_loss, self.trainer.current_epoch)
        self.logger.experiment.add_scalar(
            "Metrics/PQ_plants", self.accumulated_pq_plants, self.trainer.current_epoch)
        self.logger.experiment.add_scalar(
            "Metrics/PQ_leaves", self.accumulated_pq_leaves, self.trainer.current_epoch)

        #self.log("pqp", self.accumulated_pq_plants)
        #self.log("pql", self.accumulated_pq_leaves)

        self.accumulated_miou *= 0
        self.accumulated_pq_plants *= 0
        self.accumulated_pq_leaves *= 0
        self.val_loss *= 0

    def configure_optimizers(self):
        # OPTIMIZERS
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=self.lr[0])
        self.semantic_optimizer = torch.optim.AdamW(
            self.decoder_semseg.parameters(), lr=self.lr[1])
        self.plant_optimizer = torch.optim.AdamW(
            self.decoder_plant.parameters(), lr=self.lr[2])
        self.leaf_optimizer = torch.optim.AdamW(
            self.decoder_leaf.parameters(), lr=self.lr[3])
        # SCHEDULERS
        self.scheduler_encoder = torch.optim.lr_scheduler.StepLR(
            self.encoder_optimizer, step_size=25, gamma=0.9)
        self.scheduler_plant = torch.optim.lr_scheduler.ExponentialLR(self.plant_optimizer,gamma=0.99)
        self.scheduler_leaf= torch.optim.lr_scheduler.ExponentialLR(self.leaf_optimizer,gamma=0.99)

        return [self.encoder_optimizer, self.semantic_optimizer, self.plant_optimizer, self.leaf_optimizer], [self.scheduler_encoder, self.scheduler_plant, self.scheduler_leaf]


class Encoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.resnet = ResNet18(block='NonBottleneck1D', pretrained_on_imagenet=False)

    def forward(self, input):
        out = self.resnet(input)
        return out


class ERFNetEncoder(nn.Module):

    def __init__(self, num_classes, dropout=0.1, batch_norm=True, instance_norm=False, init=None):
        super().__init__()

        self.initial_block = DownsamplerBlock(3, 16, batch_norm, instance_norm, init)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64, batch_norm, instance_norm, init))

        DROPOUT = dropout 
        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, DROPOUT, 1, batch_norm, instance_norm, init))

        self.layers.append(DownsamplerBlock(64, 128, batch_norm, instance_norm, init))


        DROPOUT = dropout 
        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 2, batch_norm, instance_norm, init))
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 4, batch_norm, instance_norm, init))
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 8, batch_norm, instance_norm, init))
            self.layers.append(non_bottleneck_1d(128, DROPOUT, 16, batch_norm, instance_norm, init))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(
            128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = []
        output.append(self.initial_block(input))
        
        for layer in self.layers:
            output.append(layer(output[-1]))

        if predict:
            output.append(self.output_conv(output[-1]))

        return output


class DecoderSemanticSegmentation(nn.Module):
    def __init__(self, num_classes: int, dropout: float, batch_norm = True, instance_norm = False, init=None):
        super().__init__()
        self.dropout = dropout

        self.layers1 = nn.Sequential(
            UpsamplerBlock(128, 64, init),
            non_bottleneck_1d(64, self.dropout, 1, batch_norm, instance_norm, init))

        self.layers2 = nn.Sequential(
            UpsamplerBlock(64, 16, init),
            non_bottleneck_1d(16, self.dropout, 1, batch_norm, instance_norm, init))

        self.output_conv = nn.ConvTranspose2d(
            in_channels=16, out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, input):
        skip2, _, _, _, _, _, skip1, _, _, _, _, _, _, _, _, out = input

        output1 = self.layers1(out) + skip1
        output2 = self.layers2(output1) + skip2
        out = self.output_conv(output2)

        return out, [output2, output1]


class DecoderInstance(nn.Module):
    def __init__(self, dropout: float, batch_norm = True, instance_norm = False, init=None):
        super().__init__()
        self.dropout = dropout
      
        self.layers1 = nn.Sequential(
            UpsamplerBlock(128, 64, init),
            non_bottleneck_1d(64, self.dropout, 1, batch_norm, instance_norm, init),
            non_bottleneck_1d(64, self.dropout, 1, batch_norm, instance_norm, init))

        self.layers2 = nn.Sequential(
            UpsamplerBlock(64, 16, init),
            non_bottleneck_1d(16, self.dropout, 1, batch_norm, instance_norm, init),
            non_bottleneck_1d(16, self.dropout, 1, batch_norm, instance_norm, init))
        
        self.output_center = nn.ConvTranspose2d(
            in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
 
        self.output_offset = nn.ConvTranspose2d(
            in_channels=16, out_channels=2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)


    def forward(self, input):
        out, skip1, skip2 = input
        
        output1 = self.layers1(out) + skip2
        output2 = self.layers2(output1) + skip1
        centers = torch.sigmoid(self.output_center(output2))
        offsets = self.output_offset(output2)

        return centers, offsets, [output2, output1]
