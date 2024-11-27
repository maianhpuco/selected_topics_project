import argparse
import yaml

import os
from parser import parse_args, load_config
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR, train, evaluate  # Assuming your main SimCLR class and functions are here
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR Training Configuration")

    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, required=True,
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
    parser.add_argument("--config_file", type=str, default="config.yaml",
                        help="Path to the configuration YAML file (default: 'config.yaml').")

    return parser.parse_args()


def load_config(args):
    # Load default configurations from YAML file
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r") as file:
            config = yaml.safe_load(file)
    else:
        config = {}

    # Override configurations with command-line arguments
    config["dataset_path"] = args.dataset_path
    config["model"] = {
        "name": args.model_name,
        "out_dim": args.out_dim
    }
    config["batch_size"] = args.batch_size
    config["epochs"] = args.epochs
    config["lr"] = args.lr
    config["weight_decay"] = args.weight_decay
    config["log_every_n_steps"] = args.log_every_n_steps
    config["eval_every_n_epochs"] = args.eval_every_n_epochs
    config["gpu_ids"] = args.gpu_ids
    config["fp16_precision"] = args.fp16_precision

    return config

if __name__=='__main__':
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args)

    # Dataset initialization (assuming your dataset object is defined elsewhere)
    dataset = YourDatasetClass(config["dataset_path"])  # Replace with your dataset initialization logic

    # Initialize model, optimizer, and scheduler
    model = ResNetSimCLR(name=config["model"]["name"], out_dim=config["model"]["out_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=0)

    # Create SimCLR instance
    simclr = SimCLR(dataset, config)

    # Train the model
    train(simclr, model, optimizer, scheduler)

    # Evaluate the model
    evaluate_loss = evaluate(simclr, model)
    print(f"Final Validation Loss: {evaluate_loss:.4f}")
 
