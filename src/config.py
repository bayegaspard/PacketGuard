# config.py

import os
import torch

# ============================
# 1. Configuration and Setup
# ============================

# Define classes
SELECTED_CLASSES = [
    'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
    'Bot', 'FTP-Patator', 'Web Attack – Brute Force',
    'SSH-Patator', 'DoS slowloris'
]

# Maximum number of samples per class to handle imbalance
MAX_SAMPLES_PER_CLASS = 10000  # Adjust as needed

# Batch size
BATCH_SIZE = 256

# Define hyperparameter ranges for TTOPA
HYPERPARAMETERS = {
    "Learning Rate": [0.0001],# 0.001, 0.005, 0.006, 0.007, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    "Number of Steps": [100, 200, 400, 800],
    "Alpha": [0.0001], #0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "Confidence Threshold": [0.0001],# 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                             #0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
}

# Define diffusion model parameters
DIFFUSION_PARAMS = {
    "noise_steps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02
}

# Define training parameters
TRAINING_PARAMS = {
    "num_epochs_classifier": 5,  # Number of epochs for classifier
    "learning_rate_classifier": 0.0005,  # Learning rate for classifier
    "num_epochs_diffusion": 5,  # Number of epochs for diffusion model
    "learning_rate_diffusion": 1e-3  # Learning rate for diffusion model
}

# Define paths
DATA_PATH = '../../../../CICIDS2017_preprocessed.csv'  # Replace with your actual filename
RESULTS_DIR = 'results'

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Ensure the 'results' directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define model configurations
MODEL_CONFIGS = [
    {
        "description": "Default_Model",
        "hidden_sizes": [256, 128],
        "activation": "relu"
    },
    # You can add more configurations here
    # Example:
    # {
    #     "description": "Model_Deep",
    #     "hidden_sizes": [512, 256, 128],
    #     "activation": "tanh"
    # }
]
