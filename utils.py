# utils.py

import os
import seaborn as sns
import matplotlib.pyplot as plt
from config import RESULTS_DIR
import pandas as pd

def save_confusion_matrix_plot(cm, label_encoder, description=''):
    """
    Saves a confusion matrix plot.
    
    Parameters:
        cm (numpy.ndarray): Confusion matrix.
        label_encoder (LabelEncoder): Label encoder with class names.
        description (str): Description for the plot title.
    
    Returns:
        None
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label_encoder.classes_) + ['Unknown'],
                yticklabels=list(label_encoder.classes_) + ['Unknown'])
    plt.title(f'Confusion Matrix ({description})', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=45, va='center', fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{description}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved as '{plot_path}'")

def save_sensitivity_plot(sensitivity_per_feature, features, description=''):
    """
    Saves a sensitivity analysis plot.
    
    Parameters:
        sensitivity_per_feature (numpy.ndarray): Array of sensitivity scores.
        features (pd.DataFrame): DataFrame containing feature names.
        description (str): Description for the plot title.
    
    Returns:
        None
    """
    plt.figure(figsize=(20, 10))
    sns.barplot(x=list(range(len(sensitivity_per_feature))), y=sensitivity_per_feature, color='blue')
    plt.title(f"Sensitivity of Classifier to Input Features ({description})", fontsize=20)
    plt.xlabel("Feature Index", fontsize=14)
    plt.ylabel("Average Absolute Gradient", fontsize=14)
    plt.xticks(ticks=list(range(len(sensitivity_per_feature))), labels=features.columns, rotation=90, fontsize=8)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f'classifier_feature_sensitivity_{description}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Classifier feature sensitivity plot saved as '{plot_path}'")

def save_roc_curve_plot(fpr, tpr, roc_auc, optimal_threshold, description=''):
    """
    Saves the ROC curve plot.
    
    Parameters:
        fpr (array-like): False Positive Rates.
        tpr (array-like): True Positive Rates.
        roc_auc (float): ROC AUC score.
        optimal_threshold (float): Optimal threshold based on Youden's J statistic.
        description (str): Description for the plot title.
    
    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], marker='o', color='red', label='Optimal Threshold')
    plt.text(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], 
             f'  Threshold={optimal_threshold:.4f}', fontsize=12, verticalalignment='center')
    plt.title(f'ROC Curve for VarMax Thresholding ({description})', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f'roc_curve_varmax_{description}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"ROC curve plot saved as '{plot_path}'")


def append_metrics(master_df, metrics):
    """
    Appends a list of metric dictionaries to the master DataFrame.

    Parameters:
        master_df (pd.DataFrame): The master DataFrame to append metrics to.
        metrics (list of dict): List containing metric dictionaries.

    Returns:
        pd.DataFrame: Updated master DataFrame.
    """
    if not isinstance(metrics, list):
        raise TypeError("Metrics should be a list of dictionaries.")

    new_metrics_df = pd.DataFrame(metrics)
    
    # Debugging: Print shapes before concatenation
    print(f"Appending metrics with shape {new_metrics_df.shape} to master_df with shape {master_df.shape}")
    
    master_df = pd.concat([master_df, new_metrics_df], ignore_index=True)
    return master_df



