import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DEVICE

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from sklearn.metrics import classification_report, f1_score, confusion_matrix

# Inside evaluation.py
def evaluate_classifier(model, data_loader, label_encoder, description, features, master_df, step=None, attack=None):
    """
    Evaluate the classifier on the given data loader and return metrics.
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute F1 score and other metrics
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Create a new entry as a dictionary (not tuple)
    new_entry = {
        "Attack": attack,
        "Model": description,
        "Step": step,
        "F1_Score": f1
    }

    # Append the new entry to the DataFrame
    master_df = pd.concat([master_df, pd.DataFrame([new_entry])], ignore_index=True)

    print(f"F1-Score ({description}): {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    return master_df, y_true, y_pred







def save_confusion_matrix(y_true, y_pred, class_names, file_path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(file_path)  # Save the plot instead of displaying it
    plt.close()  # Close the plot to free up resources

