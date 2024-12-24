import sys
sys.path.append('src')

import os
import torch
import pandas as pd
from src.data_loader import load_and_preprocess_data
from src.defense import ttpa_improved
from src.evaluation import evaluate_classifier
from src.utils import append_metrics, save_results
from src.models import Net, VarMaxModel, DiffusionModel, FGSM, GeneticAlgorithmAttack
from src.train import train_classifier, train_diffusion_model
from src.config import MODEL_CONFIGS, TRAINING_PARAMS, DEVICE, RESULTS_DIR
from src.sensitivity_analysis import sensitivity_analysis_classifier


def main():
    print("Starting Advanced Adversarial Workflow...")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_dict = load_and_preprocess_data()
    print("Data loaded successfully.")

    features = data_dict["features"]
    label_encoder = data_dict["label_encoder"]
    num_classes = data_dict["num_classes"]
    train_loader = data_dict["train_loader"]
    test_loader = data_dict["test_loader"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]

    # Initialize a Master DataFrame to collect all metrics
    master_df = pd.DataFrame(columns=["Model", "Perturbation_Step", "Metric_Type", "Metric_Name", "Metric_Value"])

    for config in MODEL_CONFIGS:
        print(f"\n=== Training DNN Classifier: {config['description']} ===")
        
        # Train DNN Classifier
        dnn_model = Net(
            input_size=X_test.shape[1],
            num_classes=num_classes,
            hidden_sizes=config["hidden_sizes"],
            activation=config["activation"]
        ).to(DEVICE)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(dnn_model.parameters(), lr=TRAINING_PARAMS["learning_rate_classifier"])
        
        train_classifier(
            dnn_model, train_loader, criterion, optimizer,
            TRAINING_PARAMS["num_epochs_classifier"],
            test_loader, DEVICE, master_df, f"{config['description']}_DNN"
        )
        
        # Evaluate DNN on clean data
        print("\nEvaluating DNN on clean test data...")
        evaluate_classifier(dnn_model, test_loader, label_encoder, f"{config['description']}_DNN_clean", features, master_df)

        # Perform adversarial attacks (FGSM and Genetic)
        fgsm = FGSM(dnn_model, epsilon=0.03)
        genetic_attack = GeneticAlgorithmAttack(dnn_model, num_classes, max_generations=10)

        for attack, name in [(fgsm, "FGSM"), (genetic_attack, "Genetic")]:
            print(f"Applying {name} attack on DNN...")
            x_adv = attack.generate(X_test, y_test)
            evaluate_classifier(dnn_model, test_loader, label_encoder, f"{config['description']}_DNN_{name}_adv", features, master_df)

        # Train and use Diffusion Model for attack
        print(f"\n=== Training Diffusion Model for {config['description']} ===")
        diffusion_model = DiffusionModel(input_dim=X_test.shape[1]).to(DEVICE)
        diffusion_criterion = torch.nn.MSELoss()
        diffusion_optimizer = torch.optim.Adam(
            diffusion_model.parameters(), lr=TRAINING_PARAMS["learning_rate_diffusion"]
        )

        train_diffusion_model(
            diffusion_model, train_loader, diffusion_criterion, diffusion_optimizer,
            num_epochs=TRAINING_PARAMS["num_epochs_diffusion"], device=DEVICE
        )
        print("Diffusion model training completed.")

        print("Generating adversarial examples using Diffusion Model...")
        x_diff_adv = diffusion_model.generate_adversarial(X_test, t=300)
        evaluate_classifier(dnn_model, test_loader, label_encoder, f"{config['description']}_DNN_Diffusion_adv", features, master_df)

        # Apply TTOPA on DNN
        print("\nApplying TTOPA on DNN...")
        x_ttap = ttpa_improved(dnn_model, x_diff_adv, num_steps=400, learning_rate=0.001, alpha=0.9, confidence_threshold=0.7)
        evaluate_classifier(dnn_model, test_loader, label_encoder, f"{config['description']}_DNN_TTOPA", features, master_df)

        # Evaluate VarMax Classifier
        print(f"\n=== Evaluating VarMax Classifier for {config['description']} ===")
        varmax_model = VarMaxModel(dnn_model)

        # VarMax on clean data
        evaluate_classifier(varmax_model, test_loader, label_encoder, f"{config['description']}_VarMax_clean", features, master_df)

        # Adversarial attacks on VarMax (FGSM, Genetic, Diffusion)
        for attack, name in [(fgsm, "FGSM"), (genetic_attack, "Genetic"), (x_diff_adv, "Diffusion")]:
            print(f"Applying {name} attack on VarMax...")
            if name == "Diffusion":
                x_adv = x_diff_adv  # Use pre-generated diffusion adversarial examples
            else:
                x_adv = attack.generate(X_test, y_test)
            evaluate_classifier(varmax_model, test_loader, label_encoder, f"{config['description']}_VarMax_{name}_adv", features, master_df)

    # Save all results
    print("\nSaving results...")
    save_results(master_df, RESULTS_DIR)
    print("All results saved successfully.")

if __name__ == "__main__":
    main()
