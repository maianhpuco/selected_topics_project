import argparse
import yaml
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simclr import SimCLR, SimCLRDataloader, NTXentLoss, ResNetSimCLR

 
# Function to load the configuration
def load_config(config_file="configs/cnn_simclr.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config 


#  Main function to run the training and evaluation
if __name__ == '__main__':
    # Load configuration
    config = load_config()

    # Check if CUDA is available and set the device
    # device = torch.device(f"cuda:{config['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")
    device = 'cuda' if torch.cuda.is_available() and len(config['gpu_ids']) > 0 else 'cpu'
 
    print(f"Using device: {device}")
    
    from data.dataset import CheXpertDataSet  # Assuming you have this dataset class

    train_dataloader = SimCLRDataloader(
        dataset_class = CheXpertDataSet,
        data_folder =  "/project/hnguyen2/mvu9/datasets/chexpert/", 
        csv_path = './data/csv_files/train.csv', 
        batch_size=config["batch_size"], 
        num_workers=1, 
        input_shape="(224, 224, 3)", 
        s=1, 
        # csv_path=config["dataset_path"]  # Pass dataset path from config
    ).get_data_loaders()
    
    val_dataloader = SimCLRDataloader(
        dataset_class = CheXpertDataSet,
        data_folder =  "/project/hnguyen2/mvu9/datasets/chexpert/", 
        csv_path = './data/csv_files/valid.csv', 
        batch_size=config["batch_size"], 
        num_workers=1, 
        input_shape="(224, 224, 3)", 
        s=1, 
        # csv_path=config["dataset_path"]  # Pass dataset path from config
    ).get_data_loaders()
    print("test dataloader")
    batch_sample = next(iter(train_dataloader))
    inputs, labels = batch_sample
    print(inputs[0].shape, inputs[1].shape, labels.shape) 
    
    # Initialize model, optimizer, and scheduler
    model = ResNetSimCLR(
        base_model=config["model"]["base_model"], 
        out_dim=config["model"]["out_dim"]
    )
    
    # Move model to the device (GPU or CPU)

    # Create SimCLR instance
    simclr = SimCLR(train_dataloader, val_dataloader, config)
    
    # Train the model
    print("Starting training...")
    simclr.train()

    # # Evaluate the model
    # print("Evaluating the model...")
    # evaluate_loss = evaluate(simclr, model)
    # print(f"Final Validation Loss: {evaluate_loss:.4f}") 