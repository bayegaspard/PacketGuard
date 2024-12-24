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

def train_classifier(model, train_loader, criterion, optimizer, num_epochs, validation_loader, device, master_df, model_description='', label_encoder=None):
    """
    Trains the classifier model and appends training and validation metrics to master_df.

    Args:
        model (torch.nn.Module): The classifier model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.
        validation_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to train on.
        master_df (pd.DataFrame): DataFrame to append metrics.
        model_description (str): Description of the model configuration.
        label_encoder (LabelEncoder): Label encoder for class names.

    Returns:
        Tuple: Lists of training losses, validation F1 scores, accuracies, ROC AUC scores, and updated master_df.
    """
    train_losses = []
    val_f1_scores = []
    val_accuracy_scores = []
    val_roc_auc_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation on validation set
        val_f1, val_accuracy, val_roc_auc = evaluate_classifier(
            model, validation_loader, label_encoder=label_encoder,
            description=f"{model_description}_validation", features=None, master_df=None
        )
        val_f1_scores.append(val_f1)
        val_accuracy_scores.append(val_accuracy)
        val_roc_auc_scores.append(val_roc_auc)
        
        # Append metrics to master_df
        training_metrics = [{
            'Model': model_description,
            'Perturbation_Step': None,
            'Epoch': epoch + 1,
            'Metric_Type': 'Training',
            'Metric_Name': 'Loss',
            'Metric_Value': avg_loss
        }]
        
        validation_metrics = [
            {
                'Model': model_description,
                'Perturbation_Step': None,
                'Epoch': epoch + 1,
                'Metric_Type': 'Validation',
                'Metric_Name': 'F1_Score',
                'Metric_Value': val_f1
            },
            {
                'Model': model_description,
                'Perturbation_Step': None,
                'Epoch': epoch + 1,
                'Metric_Type': 'Validation',
                'Metric_Name': 'Accuracy',
                'Metric_Value': val_accuracy
            },
            {
                'Model': model_description,
                'Perturbation_Step': None,
                'Epoch': epoch + 1,
                'Metric_Type': 'Validation',
                'Metric_Name': 'ROC_AUC',
                'Metric_Value': val_roc_auc
            }
        ]
        
        # Combine all metrics
        all_metrics = training_metrics + validation_metrics
        
        # Append to master_df using the utility function
        master_df = append_metrics(master_df, all_metrics)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed. Loss: {avg_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, ROC AUC: {val_roc_auc}")
    
    return train_losses, val_f1_scores, val_accuracy_scores, val_roc_auc_scores, master_df

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
