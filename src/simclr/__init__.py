from .simclr import SimCLR, NTXentLoss, ResNetSimCLR 
from .simclr_vit import SimCLR_ver2, NTXentLoss, ViTSimCLR 

# from .loss import NTXentLoss  # Import the NTXentLoss function
from .dataset import SimCLRDataloader  # Assuming you have a dataset class for loading data

__all__ = ['SimCLR', 'SimCLR_ver2','NTXentLoss', 'SimCLRDataloader'] 