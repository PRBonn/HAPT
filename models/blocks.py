import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


##############################################################################################
#                                                                                            #
#  ERFNET blocks from https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py  #
#                                                                                            #
##############################################################################################

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated, batch_norm=False, instance_norm=False, init=None):
        super().__init__()
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm
        self.init = init

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        if self.instance_norm:
            self.in1_ = torch.nn.InstanceNorm2d(chann)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        if self.instance_norm:
            self.in2_ = torch.nn.InstanceNorm2d(chann)

        self.dropout = nn.Dropout2d(dropprob)

        # initialization
        if self.init == 'he':
            nn.init.kaiming_normal_(self.conv1x3_1.weight, mode='fan_out', nonlinearity='gelu')
            nn.init.kaiming_normal_(self.conv1x3_1.weight, mode='fan_out', nonlinearity='gelu')
            nn.init.kaiming_normal_(self.conv3x1_2.weight, mode='fan_out', nonlinearity='gelu')
            nn.init.kaiming_normal_(self.conv1x3_2.weight, mode='fan_out', nonlinearity='gelu')
            if self.batch_norm:
                nn.init.constant_(self.bn1.weight, 1)
                nn.init.constant_(self.bn1.bias, 0)
                nn.init.constant_(self.bn2.weight, 1)
                nn.init.constant_(self.bn2.bias, 0)
        elif self.init == 'xavier':
            nn.init.xavier_normal_(self.conv1x3_1.weight)
            nn.init.xavier_normal_(self.conv1x3_1.weight)
            nn.init.xavier_normal_(self.conv3x1_2.weight)
            nn.init.xavier_normal_(self.conv1x3_2.weight)
            if self.batch_norm:        
                nn.init.constant_(self.bn1.weight, 1)
                nn.init.constant_(self.bn1.bias, 0)
                nn.init.constant_(self.bn2.weight, 1)
                nn.init.constant_(self.bn2.bias, 0)
        elif self.init != "None":
            raise AttributeError("Invalid initialization")

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.gelu(output)
        output = self.conv1x3_1(output)
        if self.batch_norm:
            output = self.bn1(output)
        if self.instance_norm:
            output = self.in1_(output)
        output = F.gelu(output)

        output = self.conv3x1_2(output)
        output = F.gelu(output)
        output = self.conv1x3_2(output)
        if self.batch_norm:
            output = self.bn2(output)
        if self.instance_norm:
            output = self.in2_(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.gelu(output+input)  # +input = identity (residual connection)


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, batch_norm=False, instance_norm=False, init=None):
        super().__init__()
        self.init = init
        self.conv = nn.Conv2d(ninput, noutput-ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        if self.instance_norm:
            self.in_ = torch.nn.InstanceNorm2d(noutput)

        # initialization
        if self.init == 'he':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='gelu')
            if self.batch_norm:
                nn.init.constant_(self.bn.weight, 1)
                nn.init.constant_(self.bn.bias, 0)
        elif self.init == 'xavier':
            nn.init.xavier_normal_(self.conv.weight)
            if self.batch_norm:
                nn.init.constant_(self.bn.weight, 1)
                nn.init.constant_(self.bn.bias, 0)
        elif self.init != "None":
            raise AttributeError("Invalid initialization")

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        if self.batch_norm:
            output = self.bn(output)
        if self.instance_norm:
            output = self.in_(output)
        return F.gelu(output)

class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, init=None):
        super().__init__()
        self.init = init

        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

        # initialization
        if self.init == 'he':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='gelu')
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        elif self.init == 'xavier':
            nn.init.xavier_normal_(self.conv.weight)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        elif self.init != "None":
            raise AttributeError("Invalid initialization")

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.gelu(output)

