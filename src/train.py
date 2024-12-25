# train.py

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.autograd import Variable
from config import DEVICE
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from utils import append_metrics

# import pandas as pd
# from tqdm import tqdm
from evaluation import evaluate_classifier

# train.py

import pandas as pd
from tqdm import tqdm
from evaluation import evaluate_classifier
from utils import append_metrics  # Ensure correct import

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_classifier(model, train_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the model on the provided training data.

    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for data, labels in tqdm(train_loader, desc="Training"):
            data, labels = data.to(model.device), labels.to(model.device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")


def train_diffusion_model(diffusion_model, train_loader, criterion, optimizer, num_epochs=5, device=DEVICE):
    """
    Trains the diffusion model for adversarial example generation.
    
    Parameters:
        diffusion_model (nn.Module): The diffusion model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to run training on.
    
    Returns:
        None
    """
    diffusion_model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x_batch, _ in tqdm(train_loader, desc=f"Training Diffusion Model Epoch {epoch+1}/{num_epochs}"):
            batch_size = x_batch.shape[0]
            t = torch.randint(0, diffusion_model.noise_steps, (batch_size,), device=device).long()
            x_batch = x_batch.to(device)
            x_noisy, noise = diffusion_model.noise_data(x_batch, t)
            predicted_noise = diffusion_model(x_noisy, t)
            loss = criterion(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Diffusion Model Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
