from models.loss import BarlowTwinsLoss
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import models.blocks as blocks
import models.backbone as backbone
from pytorch_lightning.core.lightning import LightningModule
import utils.utils as utils

class BarlowTwins(LightningModule):
    def __init__(self, hparams:dict):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)
        self.model = BarlowTwinsModel(hparams)
        self.optimizer = self.configure_optimizers()
        self.step = 0
        self.loss = BarlowTwinsLoss(hparams['train']['lambd'])

    def getLoss(self, z:torch.Tensor):
        loss = self.loss(z)
        return loss

    def forward(self, x:torch.Tensor):
        y = self.model.forward(x)
        return y
   
    def training_step(self, batch, batch_idx): 
        x = batch.reshape((-1,3,224,224)) # to getfrom config
        y = self.forward(x)
        loss = self.getLoss(y)
        self.log('train:loss', loss, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.reshape((-1,3,224,224)) 
        y = self(x)
        loss = self.getLoss(y)
        self.log('val:loss', loss, prog_bar = True)
        return loss

    def test_step(self, batch, batch_idx):
        pass 

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['train']['lr-weights'])
        return [ self.optimizer ]

#######################################
# Modules
#######################################


class BarlowTwinsModel(nn.Module):
    def __init__(self, cfg:dict):
        super().__init__()
        
        # resnet creation from input size, Output stride, dropout, bn_d
        self.resnet = backbone.Backbone(cfg['data']['input_size'], 
                                          cfg['model']['output_stride'], 
                                          cfg['model']['dropout'],
                                          cfg['model']['momentum'],
                                          cfg['model']['name'] )
        
        # delete last classification layer and avg pooling: final backbone
        # to produce REPRESENTATIONS
        self.pool = nn.AdaptiveAvgPool2d(1)

        # projector is a 3 linear layer network, with cfg['model']['projector'] units
        # first two layers followed by batch norm and ReLU 
        # the output is used for the loss (EMBEDDINGS)
        sizes = [cfg['model']['embedding_size']] + list(cfg['model']['projector'])
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        
        
    def forward(self, x):
        out , skips = self.resnet(x)
        out = self.pool(out)
        out = torch.squeeze(out)
        out = self.projector(out)
        return out

    
