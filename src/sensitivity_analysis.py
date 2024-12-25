import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import shap
from config import DEVICE
from utils import append_metrics

def sensitivity_analysis_classifier(model, test_loader, label_encoder, num_classes, description, features, input_size, master_df):
    model.eval()
    gradients_list = []
    for data, labels in tqdm(test_loader, desc="Sensitivity Analysis"):
        data = data.to(DEVICE)
        inputs_valid = data.clone().detach().requires_grad_(True)
        outputs = model(inputs_valid)
        loss = torch.nn.functional.cross_entropy(outputs, labels.to(DEVICE))
        model.zero_grad()
        loss.backward()
        gradients = inputs_valid.grad.data.cpu().numpy()
        gradients_list.append(np.mean(np.abs(gradients), axis=0))
    
    avg_gradients = np.mean(gradients_list, axis=0)
    sensitivity_df = pd.DataFrame({"Feature": features, "Sensitivity": avg_gradients})
    sensitivity_df.to_csv(f"results/{description}_sensitivity.csv", index=False)



def shap_analysis(model, test_loader, feature_names):
    """
    Perform SHAP analysis on the given model and test data.

    Args:
        model (torch.nn.Module): The model to analyze.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        feature_names (list): Names of the features.

    Returns:
        ndarray: Aggregated SHAP values for features.
    """
    # Move model to the correct device
    model = model.to(DEVICE)

    # Get a batch of test data and move to the same device as the model
    data_batch = next(iter(test_loader))[0].to(DEVICE)

    # Verify input and output shapes
    print(f"Input shape: {data_batch.shape}")
    print(f"Model output shape: {model(data_batch).shape}")

    # Use SHAP DeepExplainer
    explainer = shap.DeepExplainer(model, data_batch)
    shap_values = explainer.shap_values(data_batch)

    # Check and print SHAP values shape
    for i, class_shap_values in enumerate(shap_values):
        print(f"SHAP values shape for class {i}: {class_shap_values.shape}")

    # Aggregate SHAP values across all classes
    # Sum absolute SHAP values across all output classes
    feature_shap_values = sum([abs(class_shap_values) for class_shap_values in shap_values])

    # Validate feature alignment
    if feature_shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"Aggregated SHAP values have {feature_shap_values.shape[1]} features, "
            f"but feature names imply {len(feature_names)} features."
        )

    # Save SHAP values as a CSV file
    shap_df = pd.DataFrame(feature_shap_values, columns=feature_names)
    shap_df.to_csv("results/shap_values_aggregated.csv", index=False)

    # Plot SHAP summary for features
    shap.summary_plot(feature_shap_values, feature_names=feature_names)

    return feature_shap_values