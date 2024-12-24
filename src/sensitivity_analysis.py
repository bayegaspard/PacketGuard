# # sensitivity_analysis.py

# import torch
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from config import RESULTS_DIR, DEVICE
# from utils import save_sensitivity_plot
# from torch.autograd import Variable

# def sensitivity_analysis_classifier(model, data_loader, label_encoder, num_classes, 
#                                     description='', features=None, input_size=77, master_df=None):
#     """
#     Computes and plots the sensitivity of the classifier to input features based on gradients.
    
#     Parameters:
#         model (nn.Module): Trained classification model.
#         data_loader (DataLoader): DataLoader for the test dataset.
#         label_encoder (LabelEncoder): Encoder for label classes.
#         num_classes (int): Number of known classes.
#         description (str, optional): Description of the current model configuration.
#         features (pd.DataFrame, optional): DataFrame containing feature names.
#         input_size (int, optional): Number of input features.
#         master_df (pd.DataFrame, optional): Master DataFrame to append metrics.
    
#     Returns:
#         numpy.ndarray: Array of average absolute gradients per feature.
#     """
#     model.eval()
#     sensitivity_per_feature = np.zeros(input_size)
#     sample_count = 0
#     skipped_samples = 0  # To track skipped samples due to 'Unknown' labels
    
#     for inputs_test, labels_test in data_loader:
#         # Create mask for labels within [0, num_classes - 1]
#         mask = labels_test < num_classes
#         valid_count = mask.sum().item()
#         skipped_count = (~mask).sum().item()
        
#         if valid_count == 0:
#             skipped_samples += len(labels_test)
#             continue  # Skip if no valid labels in batch
        
#         inputs_valid = inputs_test[mask]
#         labels_valid = labels_test[mask]
        
#         if inputs_valid.size(0) == 0:
#             skipped_samples += len(labels_test)
#             continue  # Skip if no valid samples
        
#         inputs_valid = Variable(inputs_valid, requires_grad=True).to(DEVICE)
#         labels_valid = labels_valid.to(DEVICE)
        
#         # Optional: Add assertion to verify label ranges
#         assert labels_valid.min() >= 0 and labels_valid.max() < num_classes, "Invalid label detected."
        
#         outputs = model(inputs_valid)
#         loss = torch.nn.CrossEntropyLoss()(outputs, labels_valid)
#         model.zero_grad()
#         loss.backward()
#         gradients = inputs_valid.grad.data.cpu().numpy()
#         sensitivity_per_feature += np.mean(np.abs(gradients), axis=0)  # Average over batch
#         sample_count += 1
#         skipped_samples += skipped_count
    
#     if sample_count == 0:
#         print("No valid samples found in the test loader for sensitivity analysis.")
#         return sensitivity_per_feature
    
#     sensitivity_per_feature /= sample_count
    
#     # Append sensitivity data to master_df
#     if master_df is not None:
#         for idx, value in enumerate(sensitivity_per_feature):
#             master_df = master_df.append({
#                 'Model': description,
#                 'Perturbation_Step': None,
#                 'Epoch': None,
#                 'Metric_Type': 'Sensitivity',
#                 'Metric_Name': f'Feature_{idx}_Sensitivity',
#                 'Metric_Value': value
#             }, ignore_index=True)
    
#     # Create a DataFrame for sensitivity and save
#     sensitivity_df = pd.DataFrame({
#         'Model': [description] * input_size,
#         'Feature': features.columns,
#         'Sensitivity': sensitivity_per_feature
#     })
#     sensitivity_df.to_csv(os.path.join(RESULTS_DIR, f'sensitivity_analysis_{description}.csv'), index=False)
#     print(f"Sensitivity analysis data saved as '{RESULTS_DIR}/sensitivity_analysis_{description}.csv'")
    
#     # Plot sensitivity per feature with adjusted font sizes
#     save_sensitivity_plot(sensitivity_per_feature, features, description)
    
#     # Log skipped samples
#     print(f"Skipped {skipped_samples} samples due to 'Unknown' labels during sensitivity analysis.")
    
#     return sensitivity_per_feature
# sensitivity_analysis.py

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import DEVICE
from utils import append_metrics

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import DEVICE
from utils import append_metrics

def sensitivity_analysis_classifier(model, test_loader, label_encoder, num_classes, description, features, input_size, master_df):
    """
    Perform sensitivity analysis by computing gradients of the loss w.r.t input features.

    Args:
        model (torch.nn.Module): Trained classifier model.
        test_loader (DataLoader): DataLoader for test data.
        label_encoder (LabelEncoder): Label encoder for class names.
        num_classes (int): Number of classes.
        description (str): Description for logging.
        features (list): List of feature names.
        input_size (int): Number of input features.
        master_df (pd.DataFrame): Master DataFrame to append metrics.

    Returns:
        pd.DataFrame: Updated master DataFrame with sensitivity metrics.
    """
    model.eval()
    gradients_list = []
    samples_analyzed = 0
    samples_skipped = 0

    for data, labels in tqdm(test_loader, desc="Sensitivity Analysis"):
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        
        # Clone and detach to ensure inputs_valid is a leaf tensor
        inputs_valid = data.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = model(inputs_valid)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()

        # Extract gradients
        if inputs_valid.grad is not None:
            grads = inputs_valid.grad.data.cpu().numpy()
            
            # Debugging: Print gradient shape
            print(f"Gradients shape before reshaping: {grads.shape}")
            
            # Ensure gradients are 2D (num_samples, num_features)
            if grads.ndim > 2:
                grads = grads.reshape(grads.shape[0], -1)
                print(f"Gradients reshaped to: {grads.shape}")
            
            gradients_list.append(grads)
            samples_analyzed += inputs_valid.size(0)
        else:
            samples_skipped += inputs_valid.size(0)

    if gradients_list:
        gradients = np.concatenate(gradients_list, axis=0)
        
        # Debugging: Print concatenated gradients shape
        print(f"Concatenated gradients shape: {gradients.shape}")
        
        # Compute average absolute gradients for each feature
        avg_gradients = np.mean(np.abs(gradients), axis=0)
        
        # Debugging: Print average gradients shape
        print(f"Average gradients shape: {avg_gradients.shape}")
        
        # Ensure avg_gradients is 1-dimensional
        if avg_gradients.ndim != 1:
            raise ValueError(f"avg_gradients is not 1-dimensional. Shape: {avg_gradients.shape}")
        
        # Ensure features list matches the number of gradients
        if len(features) != len(avg_gradients):
            raise ValueError(f"Number of features ({len(features)}) does not match number of gradients ({len(avg_gradients)}).")
        
        feature_sensitivity = pd.DataFrame({
            'Feature': features,
            'Sensitivity': avg_gradients
        }).sort_values(by='Sensitivity', ascending=False)

        print(f"Sensitivity Analysis for {description}:\n{feature_sensitivity}")

        # Append sensitivity metrics to master_df
        sensitivity_metrics = [
            {
                'Model': description.split('_')[0],
                'Perturbation_Step': description.split('_')[1] if '_' in description else None,
                'Epoch': None,
                'Metric_Type': 'Sensitivity',
                'Metric_Name': row['Feature'],
                'Metric_Value': row['Sensitivity']
            }
            for _, row in feature_sensitivity.iterrows()
        ]

        master_df = append_metrics(master_df, sensitivity_metrics)
    else:
        print("No gradients were computed. All samples were skipped.")

    if samples_skipped > 0:
        print(f"Skipped {samples_skipped} samples due to missing gradients.")

    return master_df

