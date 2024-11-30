import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import os
import sys

PROJECT_DIR = os.environ.get('PROJECT_DIR') 
sys.path.append(os.path.join(PROJECT_DIR))
sys.path.append(os.path.join(PROJECT_DIR, "src")) 

from loss.nt_xent import NTXentLoss 
import shutil
import sys
import numpy as np


class SimCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def forward(self, model, xis, xjs, n_iter):
        # Forward pass to compute loss
        ris, zis = model(xis)  # Representations and projections for xis
        rjs, zjs = model(xjs)  # Representations and projections for xjs

        # Normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        # Compute NT-Xent loss
        loss = self.nt_xent_criterion(zis, zjs)
        return loss
    
    