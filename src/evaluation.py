import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DEVICE

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from sklearn.metrics import classification_report, f1_score, confusion_matrix

def evaluate_classifier(model, dataloader, label_encoder, model_name, features, master_df, step="", attack=""):
    y_true = []
    y_pred = []
    model.eval()
    
    for data, labels in dataloader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            outputs = model(data)
        _, predictions = torch.max(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
    
    y_true_decoded = label_encoder.inverse_transform(y_true)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"F1-Score ({model_name}): {f1:.4f}")
    print(f"\nClassification Report ({model_name}):\n{classification_report(y_true_decoded, y_pred_decoded)}")
    
    new_entry = {
        "Attack": attack,
        "Model": model_name,
        "Step": step,
        "F1_Score": f1
    }
    master_df = pd.concat([master_df, pd.DataFrame([new_entry])], ignore_index=True)
    
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

