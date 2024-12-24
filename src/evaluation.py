# evaluation.py

import torch
import numpy as np
import pandas as pd
import os
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
from utils import save_confusion_matrix_plot ,append_metrics

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


def evaluate_classifier(model, test_loader, label_encoder, description, features, master_df):
    """
    Evaluates the model and returns performance metrics.
    
    Parameters:
        model (torch.nn.Module): Trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        label_encoder (LabelEncoder): Encoder for class labels.
        description (str): Description for the model or configuration.
        features (pd.DataFrame): DataFrame of feature names.
        master_df (pd.DataFrame): Master DataFrame to collect metrics.
    
    Returns:
        float, float, float: F1 score, accuracy, and ROC AUC.
    """
    model.eval()
    y_true = []
    y_pred = []
    y_proba = []  # To store probabilities for ROC AUC calculation

    # Iterate over test data
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
        target_names = label_encoder.classes_.tolist()
    else:
        # Fallback: infer class names from unique labels
        target_names = list(sorted(set(y_true)))

    # Validate that target_names is a list of strings
    target_names = [str(cn) for cn in target_names]

    # Compute classification report
    try:
        report = classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )
        print(f"Classification Report ({description}):\n{report}")
    except Exception as e:
        print(f"Error generating classification report: {e}")
        report = None
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate ROC AUC
    try:
        if len(set(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, [p[1] for p in y_proba])
        else:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception as e:
        print(f"Error calculating ROC AUC: {e}")
        roc_auc = None

    # Append to master_df
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
        master_df = append_metrics(master_df, new_metrics)

    return f1, accuracy, roc_auc




def save_confusion_matrix(y_true, y_pred, labels, output_path, title="Confusion Matrix"):
    """
    Saves a confusion matrix as a heatmap.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of label names.
        output_path (str): Path to save the confusion matrix plot.
        title (str): Title for the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Save confusion matrix as a CSV file
    csv_path = os.path.splitext(output_path)[0] + ".csv"
    cm_df.to_csv(csv_path, index=True)
    print(f"Confusion matrix saved as CSV: {csv_path}")

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved: {output_path}")
