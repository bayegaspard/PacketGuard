# utils.py

import os
import seaborn as sns
import matplotlib.pyplot as plt
from config import RESULTS_DIR
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def save_confusion_matrix_and_data(y_true, y_pred, classes, save_path, csv_path):
    """
    Save the confusion matrix plot and its data to a CSV file.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"Error: Empty predictions or labels. y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        return None

    if len(y_true) != len(y_pred):
        print(f"Error: Length mismatch. y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        return None

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    # Save plot
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Confusion matrix plot saved to {save_path}")

    # Save data to CSV
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(csv_path)
    print(f"Confusion matrix data saved to {csv_path}")
    return cm


# import numpy as np  # Add this import

# def save_confusion_matrix_to_csv(y_true, y_pred, class_labels, attack_type, step, csv_file):
#     """
#     Save confusion matrix data to a CSV file.
#     """
#     from sklearn.metrics import confusion_matrix
#     cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))
#     flattened_cm = cm.flatten()
#     header = ["Attack", "Step"] + [f"{row_label}_to_{col_label}" for row_label in class_labels for col_label in class_labels]

#     # If the CSV file doesn't exist, create it and write the header
#     if not os.path.exists(csv_file):
#         with open(csv_file, "w") as f:
#             f.write(",".join(header) + "\n")

#     # Append the confusion matrix data
#     with open(csv_file, "a") as f:
#         row = [attack_type, step] + flattened_cm.tolist()
#         f.write(",".join(map(str, row)) + "\n")



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
        master_df (pd.DataFrame or None): The master DataFrame to append metrics to. Can be None.
        metrics (list of dict): List containing metric dictionaries.

    Returns:
        pd.DataFrame: Updated master DataFrame.
    """
    if not isinstance(metrics, list):
        raise TypeError("Metrics should be a list of dictionaries.")

    new_metrics_df = pd.DataFrame(metrics)

    if master_df is None:
        print(f"Initializing master_df with metrics of shape {new_metrics_df.shape}")
        master_df = new_metrics_df
    else:
        print(f"Appending metrics with shape {new_metrics_df.shape} to master_df with shape {master_df.shape}")
        master_df = pd.concat([master_df, new_metrics_df], ignore_index=True)
    
    return master_df



def save_results(master_df, results_dir):
    """
    Saves the results from the master DataFrame and associated plots to the specified directory.

    Parameters:
        master_df (pd.DataFrame): The master DataFrame containing evaluation metrics.
        results_dir (str): Directory where results will be saved.

    Returns:
        None
    """
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Save the master DataFrame
    master_file_path = os.path.join(results_dir, 'graph_data.csv')
    master_df.to_csv(master_file_path, index=False)
    print(f"Master DataFrame saved to {master_file_path}")

    # Save individual plots for each metric type
    metric_types = master_df['Metric_Type'].unique()
    for metric_type in metric_types:
        metric_subset = master_df[master_df['Metric_Type'] == metric_type]

        if metric_type == "Sensitivity":
            # Save sensitivity analysis plot
            sensitivity_data = metric_subset.pivot_table(
                index='Metric_Name', values='Metric_Value', aggfunc='mean'
            )
            save_sensitivity_plot(
                sensitivity_data['Metric_Value'].values,
                sensitivity_data.index,
                description=metric_type
            )
        elif metric_type == "Evaluation":
            # Save evaluation metrics (confusion matrix, ROC curve, etc.)
            for model_description in metric_subset['Model'].unique():
                subset = metric_subset[metric_subset['Model'] == model_description]

                # Save confusion matrix plot
                if 'Confusion_Matrix' in subset['Metric_Name'].values:
                    cm_data = subset[subset['Metric_Name'] == 'Confusion_Matrix']['Metric_Value'].values[0]
                    save_confusion_matrix_plot(
                        cm=cm_data,
                        label_encoder=None,  # Provide the label encoder if needed
                        description=model_description
                    )

                # Save ROC curve plot
                if 'ROC_AUC' in subset['Metric_Name'].values:
                    fpr = subset[subset['Metric_Name'] == 'False_Positive_Rate']['Metric_Value'].values
                    tpr = subset[subset['Metric_Name'] == 'True_Positive_Rate']['Metric_Value'].values
                    roc_auc = subset[subset['Metric_Name'] == 'ROC_AUC']['Metric_Value'].values[0]
                    save_roc_curve_plot(
                        fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                        optimal_threshold=None,  # Provide if available
                        description=model_description
                    )
    
    print("All plots and results saved successfully.")


