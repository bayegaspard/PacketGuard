# evaluation.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import RESULTS_DIR, DEVICE
from utils import save_confusion_matrix_plot, save_roc_curve_plot

def evaluate_metrics(y_true, y_pred):
    """
    Computes evaluation metrics.
    
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        dict: Dictionary containing F1 score, precision, and recall.
    """
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"f1": f1, "precision": precision, "recall": recall}

def classify_with_unknown(model, x, top_two_threshold=0.5, varmax_threshold=0.1):
    """
    Classifies input data using Top Two Difference and VarMax.
    If the difference between top two softmax scores is less than threshold,
    uses VarMax to decide if it's an unknown class.
    
    Parameters:
        model (nn.Module): Trained classification model.
        x (torch.Tensor): Input data tensor.
        top_two_threshold (float): Threshold for Top Two Difference.
        varmax_threshold (float): Threshold for VarMax variance.
    
    Returns:
        list: List of predicted class indices or 'Unknown'.
    """
    with torch.no_grad():
        logits = model(x)
        softmax = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(softmax, 2, dim=1)
        top_diff = (top_probs[:,0] - top_probs[:,1]).cpu().numpy()
        logits_np = logits.cpu().numpy()
        
    predictions = []
    for i in range(x.size(0)):
        if top_diff[i] > top_two_threshold:
            predictions.append(top_indices[i,0].item())
        else:
            # Apply VarMax
            logit = logits_np[i]
            variance = np.var(np.abs(logit))
            if variance < varmax_threshold:
                predictions.append('Unknown')  # Unknown class
            else:
                predictions.append(top_indices[i,0].item())
    return predictions

# evaluation.py

from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score

# evaluation.py

from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score

def evaluate_classifier(model, test_loader, label_encoder, description, features, master_df):
    model.eval()
    y_true = []
    y_pred = []
    y_proba = []  # To store probabilities for ROC AUC

    for data, labels in test_loader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_proba.extend(probs.cpu().numpy())
    
    # Ensure label_encoder is provided
    if label_encoder is not None:
        target_names = label_encoder.classes_
        labels_indices = list(range(len(target_names)))  # Assuming classes are labeled from 0 to 8
    else:
        # Fallback: infer from y_true
        target_names = sorted(list(set(y_true)))
        labels_indices = sorted(list(set(y_true)))
    
    # Check for mismatch and adjust
    unique_preds = set(y_pred)
    unique_trues = set(y_true)
    total_unique = unique_preds.union(unique_trues)
    
    if len(target_names) != len(total_unique):
        print(f"Warning: Number of classes ({len(total_unique)}) does not match size of target_names ({len(target_names)}). Adjusting target_names.")
        target_names = [f"Class {i}" for i in sorted(total_unique)]
    
    report = classification_report(
        y_true, y_pred, target_names=target_names, labels=labels_indices, zero_division=0
    )
    print(f"Classification Report ({description}):\n{report}")
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate ROC AUC
    try:
        if len(labels_indices) == 2:
            roc_auc = roc_auc_score(y_true, y_proba, average='weighted')
        else:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovo', average='weighted')
    except Exception as e:
        print(f"ROC AUC Score calculation failed: {e}")
        roc_auc = None
    
    # Append to master_df if provided
    if master_df is not None:
        new_metrics = [
            {
                'Model': description.split('_')[0],
                'Perturbation_Step': description.split('_')[1] if '_' in description else None,
                'Epoch': None,
                'Metric_Type': 'Evaluation',
                'Metric_Name': 'F1_Score',
                'Metric_Value': f1
            },
            {
                'Model': description.split('_')[0],
                'Perturbation_Step': description.split('_')[1] if '_' in description else None,
                'Epoch': None,
                'Metric_Type': 'Evaluation',
                'Metric_Name': 'Accuracy',
                'Metric_Value': accuracy
            },
            {
                'Model': description.split('_')[0],
                'Perturbation_Step': description.split('_')[1] if '_' in description else None,
                'Epoch': None,
                'Metric_Type': 'Evaluation',
                'Metric_Name': 'ROC_AUC',
                'Metric_Value': roc_auc
            }
        ]
        new_metrics_df = pd.DataFrame(new_metrics)
        master_df = pd.concat([master_df, new_metrics_df], ignore_index=True)
    
    return f1, accuracy, roc_auc


