import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from src.data_loader import load_and_preprocess_data
from src.models import Net, VarMaxModel, FGSM, DiffusionModel , GeneticAlgorithmAttack
from src.utils import save_confusion_matrix_plot
from src.evaluation import evaluate_classifier
from src.defense import ttpa_improved
from src.train import train_classifier
from config import DEVICE, RESULTS_DIR

# Configuration dictionary
config = {
    "model_type": "DNN",  # Options: "DNN", "VARMAX"
    "fgsm_epsilons": [0.05, 0.1, 0.15, 0.2],
    "diffusion_steps": [10, 100, 500, 1000],
    "ttopa_steps": [10, 100, 300, 400]
}

def initialize_model(model_type, input_size, num_classes):
    if model_type == "DNN":
        return Net(input_size=input_size, num_classes=num_classes).to(DEVICE)
    elif model_type == "VARMAX":
        return VarMaxModel(input_size=input_size, num_classes=num_classes).to(DEVICE)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def main():
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    data_dict = load_and_preprocess_data()
    features = data_dict["features"]
    test_loader = data_dict["test_loader"]
    train_loader = data_dict["train_loader"]
    num_classes = data_dict["num_classes"]
    label_encoder = data_dict["label_encoder"]

    # Initialize the model based on configuration
    print("\nInitializing the model...")
    model = initialize_model(config["model_type"], len(features), num_classes)

    # Load pre-trained model or train if weights are missing
    model_path = f"default_model_{config['model_type']}.pth"
    if os.path.exists(model_path):
        print("\nLoading pre-trained model...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
    else:
        print("\nNo pre-trained model found. Training a new model...")
        train_classifier(model, train_loader, num_epochs=10, learning_rate=0.001)
        torch.save(model.state_dict(), model_path)

    # Evaluate the classifier on clean data
    print("\nEvaluating the classifier on clean data...")
    master_df = pd.DataFrame(columns=["Attack", "Model", "Step", "F1_Score"])
    master_df, y_true, y_pred = evaluate_classifier(model, test_loader, label_encoder, f"Default_Model_{config['model_type']}", features, master_df)
    # Apply Genetic Algorithm Attack
    # Apply Genetic Algorithm Attack
    print("\nApplying Genetic Algorithm Attack...")
    genetic_attack = GeneticAlgorithmAttack(model, num_classes=num_classes, max_generations=10)
    x_adv_genetic = genetic_attack.generate(data_dict["X_test"], data_dict["y_test"])

    # Debugging: Verify sizes
    print(f"x_adv_genetic size: {len(x_adv_genetic)}, y_test size: {len(data_dict['y_test'])}")

    # Validate size consistency
    assert len(x_adv_genetic) == len(data_dict["y_test"]), f"Size mismatch: x_adv_genetic ({len(x_adv_genetic)}) and y_test ({len(data_dict['y_test'])}) must match."

    # Evaluate Genetic Algorithm Attack examples
    master_df, y_true, y_pred = evaluate_classifier(
        model,
        DataLoader(TensorDataset(x_adv_genetic, torch.tensor(data_dict["y_test"]).to(DEVICE)), batch_size=256),
        label_encoder,
        f"Default_Model_{config['model_type']}_Genetic_adv",
        features,
        master_df,
        step="Genetic",
        attack="Genetic"
    )



    # FGSM Attack with epsilon variations
    print("\nRunning FGSM Attack...")
    for epsilon in config["fgsm_epsilons"]:
        print(f"\nFGSM Attack Epsilon: {epsilon}")
        fgsm = FGSM(model, epsilon=epsilon)
        x_adv_fgsm = fgsm.generate(data_dict["X_test"], data_dict["y_test"])
        master_df, y_true, y_pred = evaluate_classifier(
            model,
            DataLoader(TensorDataset(x_adv_fgsm, torch.tensor(data_dict["y_test"])), batch_size=256),
            label_encoder,
            f"Default_Model_{config['model_type']}_FGSM_adv",
            features,
            master_df,
            step=f"FGSM_Epsilon_{epsilon}",
            attack="FGSM"
        )
        save_confusion_matrix_plot(
            data_dict["y_test"],
            torch.argmax(model(x_adv_fgsm), dim=1).cpu().numpy(),
            label_encoder.classes_,
            f"results/confusion_matrix_FGSM_Epsilon_{epsilon}.png",
            f"results/confusion_matrix_FGSM_Epsilon_{epsilon}.csv"
        )
    # Inside main function after Diffusion evaluation
    print("Applying Genetic Algorithm Attack...")
    genetic_attack = GeneticAlgorithmAttack(model, num_classes=9, max_generations=10)
    x_adv_genetic = genetic_attack.generate(X_test, y_test)

    # Evaluate Genetic Attack adversarial examples
    evaluate_classifier(
        model, test_loader, label_encoder,
        description=f"{config['description']}_Genetic_adv", master_df=master_df
    )


        # TTOPA Recovery for FGSM
    print("\nRunning TTOPA for FGSM...")
    for step in config["ttopa_steps"]:
        print(f"\nTTOPA Recovery Step: {step}")
        
        # Apply TTOPA to recover from FGSM adversarial examples
        x_recovered_fgsm, grad_norms, losses = ttpa_improved(
            model, x_adv_fgsm, data_dict["y_test"], num_steps=step, learning_rate=0.01, device=DEVICE
        )
        
        # Evaluate the classifier on recovered samples
        master_df, y_true, y_pred = evaluate_classifier(
            model,
            DataLoader(TensorDataset(x_recovered_fgsm, torch.tensor(data_dict["y_test"])), batch_size=256),
            label_encoder,
            f"Default_Model_{config['model_type']}_FGSM_ttopa",
            features,
            master_df,
            step=f"TTOPA_Step_{step}",
            attack="FGSM"
        )
        
        # Save confusion matrix plot and data
        save_confusion_matrix_plot(
            data_dict["y_test"],
            torch.argmax(model(x_recovered_fgsm), dim=1).cpu().numpy(),
            label_encoder.classes_,
            f"results/confusion_matrix_FGSM_TTOPA_Step_{step}.png",
            f"results/confusion_matrix_FGSM_TTOPA_Step_{step}.csv"
        )

        print(f"TTOPA for FGSM at step {step} completed.\n")


    # Diffusion Attack with configurable steps
    print("\nRunning Diffusion Attack...")
    diffusion_attack = DiffusionModel(input_dim=len(features)).to(DEVICE)
    for step in config["diffusion_steps"]:
        print(f"\nDiffusion Attack Step: {step}")
        x_adv_diffusion = diffusion_attack.generate(data_dict["X_test"], t=step)
        master_df, y_true, y_pred = evaluate_classifier(
            model,
            DataLoader(TensorDataset(x_adv_diffusion, torch.tensor(data_dict["y_test"])), batch_size=256),
            label_encoder,
            f"Default_Model_{config['model_type']}_Diffusion_adv",
            features,
            master_df,
            step=f"Diffusion_Step_{step}",
            attack="Diffusion"
        )
        save_confusion_matrix_plot(
            data_dict["y_test"],
            torch.argmax(model(x_adv_diffusion), dim=1).cpu().numpy(),
            label_encoder.classes_,
            f"results/confusion_matrix_Diffusion_Step_{step}.png",
            f"results/confusion_matrix_Diffusion_Step_{step}.csv"
        )

    # TTOPA Recovery for Diffusion
    print("\nRunning TTOPA for Diffusion...")
    for step in config["ttopa_steps"]:
        print(f"\nTTOPA Recovery Step: {step}")
        x_recovered_diffusion, grad_norms, losses = ttpa_improved(
    model, x_adv_diffusion, y_true, num_steps=100, learning_rate=0.01, device=DEVICE
)
        master_df, y_true, y_pred = evaluate_classifier(
            model,
            DataLoader(TensorDataset(x_recovered_diffusion, torch.tensor(data_dict["y_test"])), batch_size=256),
            label_encoder,
            f"Default_Model_{config['model_type']}_Diffusion_ttopa",
            features,
            master_df,
            step=f"TTOPA_Step_{step}",
            attack="Diffusion"
        )
        save_confusion_matrix_plot(
            data_dict["y_test"],
            torch.argmax(model(x_recovered_diffusion), dim=1).cpu().numpy(),
            label_encoder.classes_,
            f"results/confusion_matrix_Diffusion_TTOPA_Step_{step}.png",
            f"results/confusion_matrix_Diffusion_TTOPA_Step_{step}.csv"
        )

    # Save evaluation results
    master_df.to_csv(os.path.join(RESULTS_DIR, f"f1_degradation_ttopa_recovery_{config['model_type']}.csv"), index=False)
    print(f"\nF1 degradation and TTOPA recovery results saved to 'f1_degradation_ttopa_recovery_{config['model_type']}.csv'.")

if __name__ == "__main__":
    main()
