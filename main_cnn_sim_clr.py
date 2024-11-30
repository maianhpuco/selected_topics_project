import argparse
import yaml
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simclr import SimCLR, SimCLRDataloader, NTXentLoss, ResNetSimCLR
from utils.train.cnn_simclr import train  # Correct import for the train function
from utils.train.cnn_simclr import evaluate  # Assuming evaluate is also in cnn_simclr.py 

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR Training Configuration")

    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, required=False,
                        help="Path to the dataset.")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="Name of the model (default: resnet50).")
    parser.add_argument("--out_dim", type=int, default=128,
                        help="Output dimension for the projection head (default: 128).")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training (default: 256).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training (default: 100).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3).")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay (default: 1e-6).")
    parser.add_argument("--log_every_n_steps", type=int, default=100,
                        help="Log training loss every n steps (default: 100).")

    # Evaluation parameters
    parser.add_argument("--eval_every_n_epochs", type=int, default=1,
                        help="Evaluate the model every n epochs (default: 1).")

    # Miscellaneous
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated list of GPU IDs to use (default: '0').")
    parser.add_argument("--fp16_precision", action="store_true",
                        help="Use mixed precision training.")
    parser.add_argument("--config_file", type=str, default="configs/cnn_simclr.yaml",
                        help="Path to the configuration YAML file (default: 'config.yaml').")

    return parser.parse_args()

# Function to load the configuration
def load_config(args):
    config = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r") as file:
            config = yaml.safe_load(file)
    else:
        print("Config file not found, using default values.")

    config["dataset_path"] = args.dataset_path or config.get("dataset_path")
    config["model"] = {
        "name": args.model_name or config.get("model", {}).get("name", "resnet50"),
        "out_dim": args.out_dim or config.get("model", {}).get("out_dim", 128)
    }
    config["batch_size"] = args.batch_size or config.get("batch_size", 256)
    config["epochs"] = args.epochs or config.get("epochs", 100)
    config["lr"] = args.lr or config.get("lr", 1e-3)
    config["weight_decay"] = args.weight_decay or config.get("weight_decay", 1e-6)
    config["log_every_n_steps"] = args.log_every_n_steps or config.get("log_every_n_steps", 100)
    config["eval_every_n_epochs"] = args.eval_every_n_epochs or config.get("eval_every_n_epochs", 1)
    config["gpu_ids"] = args.gpu_ids or config.get("gpu_ids", "0")
    config["fp16_precision"] = args.fp16_precision or config.get("fp16_precision", False)

    return config

# Main function to run the training and evaluation
if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args)

    # Check if CUDA is available and set the device
    device = torch.device(f"cuda:{args.gpu_ids}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    from data.dataset import CheXpertDataSet

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

    # Assuming SimCLRDataloader returns a tuple with (train_loader, valid_loader)

    # Example batch from the train loader (to check the data loading)
    # batch_sample = next(iter(train_dataloader))
    # inputs, _ = batch_sample
    # print(inputs[0].shape, inputs[1].shape)  # Check shape of input tensors

    # Initialize model, optimizer, and scheduler
    model = ResNetSimCLR(
        base_model=config["model"]["name"], 
        out_dim=config["model"]["out_dim"])
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=0)

    # Move model to the device (GPU or CPU)
    model = model.to(device)
    
    dataset = train_dataloader 
    # Create SimCLR instance
    simclr = SimCLR(dataset, config)
    simclr.train() 

    # # Train the model
    # print("Starting training...")
    # train(simclr, model, optimizer, scheduler)

    # # Evaluate the model
    # print("Evaluating the model...")
    # evaluate_loss = evaluate(simclr, model)
    # print(f"Final Validation Loss: {evaluate_loss:.4f}")
