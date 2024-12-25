import os
import torch

# Path to the dataset
DATA_PATH = '../../../../CICIDS2017_preprocessed.csv'  # Replace with the actual path to your dataset

# Class definitions for filtering
SELECTED_CLASSES = [
    'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
    'Bot', 'FTP-Patator', 'Web Attack – Brute Force',
    'SSH-Patator', 'DoS slowloris'
] 

# Configuration parameters
MAX_SAMPLES_PER_CLASS = 10000
BATCH_SIZE = 256
RESULTS_DIR = 'results'

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model configurations
MODEL_CONFIGS = [
    {
        "description": "Default_Model",
        "hidden_sizes": [256, 128],
        "activation": "relu"
    }
]

# Diffusion model parameters
DIFFUSION_PARAMS = {
    "noise_steps": [10,100,500,1000],
    "beta_start": 1e-4,
    "beta_end": 0.02,
}

