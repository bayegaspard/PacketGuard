import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DEVICE

import pandas as pd
from sklearn.metrics import classification_report, f1_score

def evaluate_classifier(model, data_loader, label_encoder, model_name, features, master_df, step=None, attack=None):
    y_true, y_pred = [], []

    # Evaluate the model on the dataset
    for data, labels in data_loader:
        data, labels = data.to(model.device), labels.to(model.device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    # Compute F1-score
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"F1-Score ({model_name}): {f1:.4f}")

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(f"\nClassification Report ({model_name}):\n{report}")

    # Add results to the master DataFrame
    new_entry = pd.DataFrame([{
        "Attack": attack,
        "Model": model_name,
        "Step": step,
        "F1_Score": f1
    }])
    master_df = pd.concat([master_df, new_entry], ignore_index=True)

    return master_df



def save_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
    plt.savefig(output_path)
    plt.close()
