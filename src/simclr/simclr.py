import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm  
import os
import sys
import numpy as np 
import timm

import shutil
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
 
# Check if Apex is available for mixed precision training
apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp
    apex_support = True
except ImportError:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

# Set the manual seed for reproducibility
torch.manual_seed(0)
# Function to save the config file to the checkpoint directory
def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./configs/cnn_simclr.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))
        
        
         
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size) 

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=True),  # Default to BatchNorm2d
            "resnet50": models.resnet50(pretrained=True)
        }

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim) 
        
    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except KeyError:
            raise ValueError("Invalid model name. Choose 'resnet18' or 'resnet50'.")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x   

class SimCLR(object):

    def __init__(self,train_dataloader, val_dataloader , config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss
     
    def train(self):
        train_loader, valid_loader = self.train_dataloader, self.val_dataloader

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        if self.config['n_gpu'] > 1:
            device_n = len(eval(self.config['gpu_ids']))
            model = torch.nn.DataParallel(model, device_ids=range(device_n))

        model = self._load_pre_trained_weights(model)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=eval(self.config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0, last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2', keep_batchnorm_fp32=True)

        # model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        model_name = self.config["model"]["base_model"]  # Assuming base_model holds the name (e.g., "resnet18" or "resnet50")
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, f'checkpoints_{model_name}', timestamp)
 
        print("model checkpoints folder: ", model_checkpoints_folder)
        # Make sure the folder exists
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder) 
        
        # Save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            start_time = time.time()  # Record start time for the epoch
            
            epoch_loss = 0  # Initialize the loss accumulator for the epoch

            # Use tqdm for progress bar
            total_batches = len(train_loader)
            with tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch_counter + 1}/{self.config['epochs']}") as t:
                for batch_idx, ((xis, xjs), label) in t:
                    optimizer.zero_grad()

                    xis = xis.to(self.device)
                    xjs = xjs.to(self.device)

                    loss = self._step(model, xis, xjs, n_iter)
                    epoch_loss += loss.item()  # Accumulate loss

                    # Update tqdm with loss information
                    t.set_postfix(loss=loss.item())

                    # if n_iter % self.config['print_every_n_batches'] == 0:  # Print results after certain batches
                    #     print(f"Epoch [{epoch_counter + 1}/{self.config['epochs']}], Batch [{batch_idx + 1}/{total_batches}], Loss: {loss.item():.4f}")

                    if n_iter % self.config['log_every_n_steps'] == 0:
                        self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                    if apex_support and self.config['fp16_precision']:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    optimizer.step()
                    n_iter += 1

            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / total_batches
            print(f"Epoch [{epoch_counter + 1}/{self.config['epochs']}], Average Loss: {avg_epoch_loss:.4f}")

            # Validate the model if requested
            # Inside your training loop, modify the model saving code:
            # if valid_loss < best_valid_loss:
            #     # Save the model weights with the name of the model architecture and the epoch number in the filename
            #     best_valid_loss = valid_loss
            #     epoch_number = epoch_counter + 1  # Epoch number (1-based index)
            #     model_filename = f'epoch_{epoch_number:03d}_model.pth'
                
            #     # Save the model with the updated filename and folder
            #     torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, model_filename))
            #     print(f'Model saved as {model_filename} in {model_checkpoints_folder}.') 
                
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # Save the model weights
                    best_valid_loss = valid_loss
                    epoch_number = epoch_counter + 1  # Epoch number (1-based index)
                    model_filename = f'epoch{epoch_number:03d}_model.pth' 
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder,model_filename))
                    print(f'Model saved as {model_filename} in {model_checkpoints_folder}.')  

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # Warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)  

            # Record and print time taken for the epoch
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Time taken for Epoch {epoch_counter + 1}: {epoch_time:.2f} seconds") 

    def _load_pre_trained_weights(self, model):
        if self.config.get('fine_tune_from'):  # Check if fine_tune_from is set
            try:
                checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
                state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
                model.load_state_dict(state_dict)
                print("Loaded pre-trained model with success.")
            except FileNotFoundError:
                print(f"No pre-trained weights found in {checkpoints_folder}. Training from scratch.")
        else:
            print("No pre-trained weights specified. Training from scratch.")
        return model
    

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs), label in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
    
    
    