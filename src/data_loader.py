import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import (
    DATA_PATH, SELECTED_CLASSES,
    MAX_SAMPLES_PER_CLASS, BATCH_SIZE, DEVICE
)

def load_and_preprocess_data():
    # Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"The dataset file '{DATA_PATH}' was not found.")

    data = pd.read_csv(DATA_PATH)

    # Drop unnecessary columns
    columns_to_drop = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
        'Destination Port', 'Timestamp'
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(existing_columns_to_drop, axis=1, inplace=True)

    # Handle missing and infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Filter data to include only the selected classes
    data = data[data['Label'].isin(SELECTED_CLASSES)]

    # Limit the number of samples per class to handle imbalance
    data = data.groupby('Label').apply(
        lambda x: x.sample(n=min(len(x), MAX_SAMPLES_PER_CLASS), random_state=42)
    ).reset_index(drop=True)

    # Print class distribution after limiting samples
    print("\nClass distribution after limiting samples per class:")
    print(data['Label'].value_counts())

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['Label'])
    num_classes = len(label_encoder.classes_)
    print("\nEncoded Classes:", label_encoder.classes_)

    # Convert features to numeric
    features_df = data.drop('Label', axis=1)  # Fix: Define features_df here
    features_df = features_df.apply(pd.to_numeric, errors='coerce')
    features_df.fillna(0, inplace=True)

    # Extract feature names as a list
    features = features_df.columns.tolist()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # Split into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return {
        "features": features,
        "features_df": features_df,  # Now correctly defined
        "scaler": scaler,
        "label_encoder": label_encoder,
        "num_classes": num_classes,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "X_train": X_train_tensor,
        "X_test": X_test_tensor,
        "y_train": y_train_tensor,
        "y_test": y_test_tensor
    }
