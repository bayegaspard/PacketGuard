import sys
sys.path.append('src') 
import torch
import torch.optim as optim
import pandas as pd
from src.data_loader import load_and_preprocess_data
from src.defense import ttpa_improved
from src.evaluation import evaluate_metrics
from src.utils import append_metrics
from src.models import Net
from src.train import *
from src.config import *
from src.models import *
from src.sensitivity_analysis import *
def main():
    print("Starting PacketGuard Workflow...")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_dict = load_and_preprocess_data()
    print("Data loaded successfully.")
    
    features = data_dict["features"]  # List of feature names (length should be 77)
    scaler = data_dict["scaler"]
    label_encoder = data_dict["label_encoder"]
    num_classes = data_dict["num_classes"]
    train_loader = data_dict["train_loader"]
    test_loader = data_dict["test_loader"]
    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    
    # Debugging: Verify features list
    print(f"Number of features passed to main: {len(features)}")
    print(f"First 10 feature names: {features[:10]}")
    
    # Initialize a Master DataFrame to collect all metrics
    print("Initializing Master DataFrame...")
    master_df = pd.DataFrame(columns=[
        'Model', 'Perturbation_Step', 'Epoch', 
        'Metric_Type', 'Metric_Name', 'Metric_Value'
    ])
    
    # ============================
    # 1. Define and Train the Classifier
    # ============================
    
    for config in MODEL_CONFIGS:
        print(f"\n=== Training Classifier: {config['description']} ===")
        model = Net(
            input_size=X_train.shape[1],
            num_classes=num_classes,
            hidden_sizes=config['hidden_sizes'],
            activation=config['activation']
        ).to(DEVICE)
        
        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS["learning_rate_classifier"])
        
        # Train the classifier with test_loader as validation_loader and append metrics to master_df
        print("Starting classifier training...")
        train_losses, val_f1_scores, val_accuracy_scores, val_roc_auc_scores, master_df = train_classifier(
            model, train_loader, criterion, optimizer, num_epochs=TRAINING_PARAMS["num_epochs_classifier"],
            validation_loader=test_loader, device=DEVICE, master_df=master_df,
            model_description=config['description'], label_encoder=label_encoder  # Pass label_encoder here
        )
        print("Classifier training completed.")
        
        # ============================
        # 2. Train the Diffusion Model
        # ============================
        print(f"\n=== Training Diffusion Model for {config['description']} ===")
        diffusion_model = DiffusionModel(input_dim=X_train.shape[1]).to(DEVICE)
        diffusion_criterion = torch.nn.MSELoss()
        diffusion_optimizer = torch.optim.Adam(diffusion_model.parameters(), 
                                               lr=TRAINING_PARAMS["learning_rate_diffusion"])
        
        print("Starting diffusion model training...")
        train_diffusion_model(
            diffusion_model, train_loader, diffusion_criterion, diffusion_optimizer,
            num_epochs=TRAINING_PARAMS["num_epochs_diffusion"], device=DEVICE
        )
        print("Diffusion model training completed.")
        
        # ============================
        # 3. Evaluate on Test Data (Clean)
        # ============================
        print(f"\n=== Evaluating on Clean Test Data for {config['description']} ===")
        f1, accuracy, roc_auc = evaluate_classifier(
            model, test_loader, label_encoder, 
            description=f"{config['description']}_clean", 
            features=features, master_df=master_df
        )
        print("Evaluation on clean test data completed.")
        
        # ============================
        # 4. Sensitivity Analysis on Clean Test Data
        # ============================
        print("Starting sensitivity analysis on clean test data...")
        master_df = sensitivity_analysis_classifier(
            model,
            test_loader,
            label_encoder,
            num_classes,  # Number of known classes
            description=f"{config['description']}_clean",
            features=features,  # Correctly pass feature names
            input_size=X_train.shape[1],
            master_df=master_df
        )
        print("Sensitivity analysis completed.")
        
        # ============================
        # 5. Generate Adversarial Examples and Evaluate
        # ============================
        print(f"\n=== Generating and Evaluating Adversarial Examples for {config['description']} ===")
        
        y_adv_pred = []
        x_adv_list = []  # To store all adversarial examples
        model.eval()  # Ensure the model is in evaluation mode
        
        print("Generating adversarial examples and evaluating...")
        for data_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            batch_size = data.size(0)
    
            # Generate adversarial examples at a lower diffusion step (less perturbation)
            step = 300  # Adjusted diffusion step
            with torch.no_grad():
                x_adv = diffusion_model.generate_adversarial(data, step)
                x_adv_list.append(x_adv.cpu())  # Collect adversarial examples
    
            # Evaluate on adversarial examples
            with torch.no_grad():
                outputs_adv = model(x_adv)
                _, preds_adv = torch.max(outputs_adv.data, 1)
                y_adv_pred.extend(preds_adv.cpu().numpy())
    
        # Concatenate all adversarial examples
        if x_adv_list:
            x_adv_full = torch.cat(x_adv_list, dim=0)
            print(f"Total adversarial examples collected: {x_adv_full.shape[0]}")
        else:
            raise ValueError("No adversarial examples were generated.")
    
        # Ensure x_adv_full has the same number of samples as y_test
        if x_adv_full.shape[0] != len(y_test):
            raise ValueError(f"x_adv_full has {x_adv_full.shape[0]} samples, but y_test has {len(y_test)} samples.")
        
        # ============================
        # 6. Apply TTOPA and Evaluate
        # ============================
        print(f"\n=== Applying TTOPA and Evaluating for {config['description']} ===")
        # Define hyperparameter combinations for TTOPA
        hyperparams = HYPERPARAMETERS.copy()
        
        for param_name, param_values in hyperparams.items():
            print(f"\n=== Testing TTOPA Hyperparameter: {param_name} ===")
            metrics_ttap = {"TTOPA": []}
            
            for value in param_values:
                print(f"\n--- Testing {param_name} = {value} ---")
                # Apply TTOPA with current hyperparameter
                if param_name == "Learning Rate":
                    x_ttap = ttpa_improved(
                        model, 
                        x_adv_full,  # Use the full adversarial dataset
                        num_steps=400,  # Example value, can be parameterized
                        learning_rate=value, 
                        alpha=0.9, 
                        confidence_threshold=0.7
                    )
                elif param_name == "Number of Steps":
                    x_ttap = ttpa_improved(
                        model, 
                        x_adv_full, 
                        num_steps=value, 
                        learning_rate=0.001, 
                        alpha=0.9, 
                        confidence_threshold=0.7
                    )
                elif param_name == "Alpha":
                    x_ttap = ttpa_improved(
                        model, 
                        x_adv_full, 
                        num_steps=400, 
                        learning_rate=0.001, 
                        alpha=value, 
                        confidence_threshold=0.7
                    )
                elif param_name == "Confidence Threshold":
                    x_ttap = ttpa_improved(
                        model, 
                        x_adv_full, 
                        num_steps=400, 
                        learning_rate=0.001, 
                        alpha=0.9, 
                        confidence_threshold=value
                    )
                else:
                    print(f"Unknown hyperparameter: {param_name}")
                    continue
    
                # Validate the number of samples in x_ttap
                if x_ttap.shape[0] != x_adv_full.shape[0]:
                    raise ValueError(f"TTOPA output has {x_ttap.shape[0]} samples, expected {x_adv_full.shape[0]} samples.")
    
                # Evaluate on TTOPA adapted adversarial examples
                with torch.no_grad():
                    outputs_ttap = model(x_ttap)
                    _, preds_ttap = torch.max(outputs_ttap.data, 1)
                    y_ttap_pred = preds_ttap.cpu().numpy()
    
                # Validate the number of predictions
                if len(y_ttap_pred) != len(y_test):
                    raise ValueError(f"y_ttap_pred has {len(y_ttap_pred)} samples, but y_test has {len(y_test)} samples.")
    
                # Evaluate metrics
                ttap_metrics = evaluate_metrics(y_test, y_ttap_pred)
                
                # Append metrics to master_df
                ttap_metric_entry = [
                    {
                        'Model': config['description'],
                        'Perturbation_Step': f'TTOPA_{param_name}',
                        'Epoch': None,
                        'Metric_Type': 'TTOPA',
                        'Metric_Name': f'{param_name}_F1_Score',
                        'Metric_Value': ttap_metrics['f1']
                    },
                    {
                        'Model': config['description'],
                        'Perturbation_Step': f'TTOPA_{param_name}',
                        'Epoch': None,
                        'Metric_Type': 'TTOPA',
                        'Metric_Name': f'{param_name}_Precision',
                        'Metric_Value': ttap_metrics['precision']
                    },
                    {
                        'Model': config['description'],
                        'Perturbation_Step': f'TTOPA_{param_name}',
                        'Epoch': None,
                        'Metric_Type': 'TTOPA',
                        'Metric_Name': f'{param_name}_Recall',
                        'Metric_Value': ttap_metrics['recall']
                    }
                ]
                master_df = append_metrics(master_df, ttap_metric_entry)
        
        # ============================
        # Save the Master DataFrame
        # ============================
        if not master_df.empty:
            master_df.to_csv(os.path.join(RESULTS_DIR, 'graph_data.csv'), index=False)
            print("\nAll metrics saved to 'results/graph_data.csv'")
        else:
            print("\nMaster DataFrame is empty. No metrics to save.")
    
    print("\n=== All Experiments Completed ===")
    print("All results and data have been saved in the 'results/' directory.")

if __name__ == "__main__":
    main()
