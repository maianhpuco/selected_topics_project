import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
import shutil
import sys
import numpy as np

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
    
except ImportError:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

torch.manual_seed(0)

def save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


def train_one_epoch(simclr, model, optimizer, train_loader, device, writer, n_iter):
    model.train()
    epoch_loss = 0.0

    for (xis, xjs) in train_loader:
        optimizer.zero_grad()

        xis = xis.to(device)
        xjs = xjs.to(device)

        # Compute loss
        loss = simclr.forward(model, xis, xjs, n_iter)

        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()

        # Logging
        epoch_loss += loss.item()
        if n_iter % simclr.config['log_every_n_steps'] == 0:
            writer.add_scalar('train_loss', loss.item(), global_step=n_iter)

        n_iter += 1

    return n_iter, epoch_loss / len(train_loader)


def train(simclr, model, optimizer, scheduler):
    train_loader, _ = simclr.dataset.get_data_loaders()
    model = model.to(simclr.device)
    n_iter = 0

    for epoch in range(simclr.config['epochs']):
        print(f"Epoch {epoch + 1}/{simclr.config['epochs']}")

        # Train for one epoch
        n_iter, avg_epoch_loss = train_one_epoch(
            simclr, model, optimizer, train_loader, simclr.device, simclr.writer, n_iter
        )

        print(f"Epoch [{epoch + 1}/{simclr.config['epochs']}], Loss: {avg_epoch_loss:.4f}")

        # Scheduler step
        if scheduler:
            scheduler.step()
            simclr.writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=n_iter)

    print("Training complete.")


def evaluate(simclr, model):
    """
    Evaluates the model on the validation dataset.

    Args:
        simclr: SimCLR instance for handling forward and evaluation.
        model: The model to evaluate.

    Returns:
        avg_valid_loss: Average validation loss.
    """
    _, valid_loader = simclr.dataset.get_data_loaders()
    model = model.to(simclr.device)
    model.eval()

    valid_loss = 0.0
    with torch.no_grad():
        for (xis, xjs) in valid_loader:
            xis = xis.to(simclr.device)
            xjs = xjs.to(simclr.device)

            # Compute loss
            loss = simclr.forward(model, xis, xjs, n_iter=None)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    print(f"Validation Loss: {avg_valid_loss:.4f}")
    simclr.writer.add_scalar('validation_loss', avg_valid_loss)
    return avg_valid_loss
 