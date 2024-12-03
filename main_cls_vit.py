import argparse
import yaml
import torch
import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils', 'train'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data')) 

from utils.train.cls import train  # Define these as per earlier instructions
from dataset import CheXpertDataSet  # Replace with your dataset class
import time 
import numpy as np 
import torch.nn as nn
from torchvision import models
import timm 
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, hamming_loss, classification_report
from tqdm import tqdm 
def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate metrics for multi-label classification, including AUC-ROC and a detailed classification report.
    """
    metrics = {}

    # Compute AUC-ROC for each class
    num_classes = y_true.shape[1]
    class_wise_auc = {}
    for class_idx in range(num_classes):
        try:
            auc = roc_auc_score(y_true[:, class_idx], y_prob[:, class_idx])
        except ValueError:  # Handle cases where AUC cannot be computed
            auc = None
        class_wise_auc[f"Class {class_idx}"] = auc

    # Compute macro-average AUC-ROC
    try:
        macro_auc_roc = roc_auc_score(y_true, y_prob, average='macro')
    except ValueError:
        macro_auc_roc = None

    # Other metrics
    metrics.update({
        "Macro AUC-ROC": macro_auc_roc,
        "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Hamming Loss": hamming_loss(y_true, y_pred)
    })

    # Add class-wise AUC-ROC to metrics
    metrics["Class-wise AUC-ROC"] = class_wise_auc

    # Generate a classification report
    try:
        class_report = classification_report(
            y_true,
            y_pred,
            target_names=[f"Class {i}" for i in range(num_classes)],
            zero_division=0,
            output_dict=True
        )
    except ValueError:
        class_report = "Cannot generate classification report. Invalid inputs."

    return metrics, class_report
  
  

class MultiLabelModel(nn.Module):
    def __init__(self, base_model, num_classes, ssl_checkpoint=None):
        super(MultiLabelModel, self).__init__()
        
        if base_model == "resnet18":
            self.model = models.resnet18(pretrained=False)  # Use pretrained=False since SSL checkpoint will be loaded
        elif base_model == "resnet50":
            self.model = models.resnet50(pretrained=False)
        elif base_model == "vit_base_patch16_224":
            self.model = timm.create_model("vit_base_patch16_224", pretrained=False)  # Exclude FC layer for SSL
        else:
            raise ValueError("Unsupported base model. Choose from ['resnet18', 'resnet50', 'vit_base_patch16_224']")
        
        # Load SSL checkpoint if provided
        if ssl_checkpoint:
            self._load_ssl_checkpoint(ssl_checkpoint)
        
        # Replace the final classification layer
        if base_model in ["resnet18", "resnet50"]:
            num_ftrs = self.model.fc.in_features  # Get features of the final FC layer
            self.model.fc = nn.Linear(num_ftrs, num_classes)  # Replace FC layer for multi-label output
        elif base_model == "vit_base_patch16_224":
            num_ftrs = self.model.num_features  # Get features from ViT
            self.fc = nn.Linear(num_ftrs, num_classes)  # Add classification head
            self.model.head = nn.Identity()  # Remove existing head for ViT

    def forward(self, x):
        if hasattr(self, "fc"):  # For ViT models
            features = self.model(x)
            return self.fc(features)
            
        return self.model(x)

    def _load_ssl_checkpoint(self, checkpoint_path):
        """
        Load SSL checkpoint weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Modify key names if necessary
        modified_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # Remove "module." if it exists
            modified_state_dict[new_key] = value

        # Load the state dict
        self.model.load_state_dict(modified_state_dict, strict=False)
        print(f"Loaded SSL checkpoint weights from {checkpoint_path}") 
    
     


if __name__ == '__main__':
    # Set device
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    MODEL_NAME = 'vit_base_patch16_224'
    
    base_model = MODEL_NAME
    save_dir = './weight_cls/'
    
    if MODEL_NAME == 'resnet18':
        WEIGHT_PATH =  'runs/Nov30_20-03-40_compute-0-8.local/checkpoints_resnet18/2024-11-30_20-03/epoch046_model.pth'
    elif MODEL_NAME == 'resnet50':
        WEIGHT_PATH =  'runs/Nov30_20-19-46_compute-0-8.local/checkpoints_resnet50/2024-11-30_20-19/epoch046_model.pth'
    elif MODEL_NAME == 'vit_base_patch16_224':
        WEIGHT_PATH =  'runs/Dec01_19-39-46_compute-0-8.local/checkpoints_vit_base_patch16_224/2024-12-01_19-39/epoch027_model.pth'
    
    ssl_checkpoint_path = WEIGHT_PATH 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data paths
    data_folder = "/project/hnguyen2/mvu9/datasets/chexpert/"
    train_csv_path = './data/csv_files/train.csv'
    valid_csv_path = './data/csv_files/valid.csv'
    test_csv_path  = './data/csv_files/test.csv'

    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load datasets
    train_dataset = CheXpertDataSet(data_folder, train_csv_path, transform=data_transforms, policy="ones")
    val_dataset = CheXpertDataSet(data_folder, valid_csv_path, transform=data_transforms, policy="ones")
    test_dataset = CheXpertDataSet(data_folder, test_csv_path, transform=data_transforms, policy="ones")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=False)
    val_loader  = DataLoader(val_dataset,    batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=False)
    test_loader = DataLoader(test_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=False)

    

    # Initialize the model
    num_classes = 14

    model = MultiLabelModel(
        base_model=base_model, 
        num_classes=num_classes, 
        ssl_checkpoint=ssl_checkpoint_path)

    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)  



    # model = ResNetMultiLabel(num_classes).to(device)
    loss_criteria = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.strftime("%Y-%m-%d_%H-%M")
    checkpoint_dir = f"./weight_cls/vit_checkpoints_{start_time}" 

    trained_model = train(
        model, 
        train_loader, 
        val_loader, 
        device, 
        loss_criteria, 
        optimizer, 
        NUM_EPOCHS, 
        checkpoint_dir=checkpoint_dir) 
    
    # Calculate additional metrics
    y_true = []
    y_pred = []
    y_prob = []

    trained_model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Validating", leave=False)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = trained_model(images)

            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > 0.5).astype(float)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions)
            y_prob.extend(probabilities)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = calculate_metrics(y_true, y_pred, y_prob)
     
    file_path = os.path.join(save_dir, f"{MODEL_NAME}_{start_time}.yaml")

    # Save metrics to a YAML file
    with open(file_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)

    print(f"Metrics saved to {file_path}") 
    print(metrics[0])
        
    