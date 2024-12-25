import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from torch.utils.data import DataLoader, TensorDataset
from src.data_loader import load_and_preprocess_data
from src.models import Net, FGSM, GeneticAlgorithmAttack, DiffusionModel
from src.sensitivity_analysis import sensitivity_analysis_classifier, shap_analysis
from src.utils import save_confusion_matrix_and_data
from src.evaluation import evaluate_classifier
from src.defense import ttpa_improved
from src.train import train_classifier
from config import DEVICE, RESULTS_DIR

def main():
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    data_dict = load_and_preprocess_data()
    features = data_dict["features"]
    test_loader = data_dict["test_loader"]
    train_loader = data_dict["train_loader"]
    num_classes = data_dict["num_classes"]
    label_encoder = data_dict["label_encoder"]

    # Initialize the model
    print("\nInitializing the model...")
    dnn_model = Net(input_size=len(features), num_classes=num_classes).to(DEVICE)

    # Load pre-trained model or train if weights are missing
    model_path = "default_model.pth"
    if os.path.exists(model_path):
        print("\nLoading pre-trained model...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        dnn_model.load_state_dict(checkpoint)
    else:
        print("\nNo pre-trained model found. Training a new model...")
        train_classifier(dnn_model, train_loader, num_epochs=10, learning_rate=0.001)
        torch.save(dnn_model.state_dict(), model_path)

    # Evaluate the classifier on clean data
    print("\nEvaluating the classifier on clean data...")
    master_df = pd.DataFrame(columns=["Attack", "Model", "Step", "F1_Score"])
    confusion_data = []
    master_df, y_true, y_pred = evaluate_classifier(dnn_model, test_loader, label_encoder, "Default_Model", features, master_df)
    save_confusion_matrix_and_data(y_true, y_pred, label_encoder.classes_, "results/confusion_matrix_Default_Model.png", "results/confusion_matrix_Default_Model.csv")

    # FGSM Attack
    print("\nRunning FGSM Attack...")
    fgsm = FGSM(dnn_model, epsilon=0.1)
    for step in [10, 100, 500, 1000]:
        print(f"\nFGSM Attack Step: {step} (steps ignored for FGSM, using default epsilon)")
        x_adv = fgsm.generate(data_dict["X_test"], data_dict["y_test"])
        master_df, y_true, y_pred = evaluate_classifier(
            dnn_model,
            DataLoader(TensorDataset(x_adv, torch.tensor(data_dict["y_test"])), batch_size=256),
            label_encoder,
            "Default_Model_FGSM_adv",
            features,
            master_df,
            step=f"FGSM_Step_{step}",
            attack="FGSM"
        )
        save_confusion_matrix_and_data(y_true, y_pred, label_encoder.classes_, f"results/confusion_matrix_FGSM_Step_{step}.png", f"results/confusion_matrix_FGSM_Step_{step}.csv")

    # TTOPA Recovery for FGSM
    print("\nRunning TTOPA for FGSM...")
    for step in [10, 100, 300, 400]:
        print(f"\nTTOPA Recovery Step: {step}")
        x_recovered_fgsm = ttpa_improved(dnn_model, x_adv)
        master_df, y_true, y_pred = evaluate_classifier(
            dnn_model,
            DataLoader(TensorDataset(x_recovered_fgsm, torch.tensor(data_dict["y_test"])), batch_size=256),
            label_encoder,
            "Default_Model_FGSM_ttopa",
            features,
            master_df,
            step=f"TTOPA_Step_{step}",
            attack="FGSM"
        )
        save_confusion_matrix_and_data(y_true, y_pred, label_encoder.classes_, f"results/confusion_matrix_FGSM_TTOPA_Step_{step}.png", f"results/confusion_matrix_FGSM_TTOPA_Step_{step}.csv")

    # Diffusion Attack
    print("\nRunning Diffusion Attack...")
    diffusion_attack = DiffusionModel(input_dim=len(features)).to(DEVICE)
    for step in [10, 100, 500, 1000]:
        print(f"\nDiffusion Attack Step: {step}")
        x_adv_diffusion = diffusion_attack.generate(data_dict["X_test"], t=step)
        master_df, y_true, y_pred = evaluate_classifier(
            dnn_model,
            DataLoader(TensorDataset(x_adv_diffusion, torch.tensor(data_dict["y_test"])), batch_size=256),
            label_encoder,
            "Default_Model_Diffusion_adv",
            features,
            master_df,
            step=f"Diffusion_Step_{step}",
            attack="Diffusion"
        )
        save_confusion_matrix_and_data(y_true, y_pred, label_encoder.classes_, f"results/confusion_matrix_Diffusion_Step_{step}.png", f"results/confusion_matrix_Diffusion_Step_{step}.csv")

    # TTOPA Recovery for Diffusion
    print("\nRunning TTOPA for Diffusion...")
    for step in [10, 100, 300, 400]:
        print(f"\nTTOPA Recovery Step: {step}")
        x_recovered_diffusion = ttpa_improved(dnn_model, x_adv_diffusion)
        master_df, y_true, y_pred = evaluate_classifier(
            dnn_model,
            DataLoader(TensorDataset(x_recovered_diffusion, torch.tensor(data_dict["y_test"])), batch_size=256),
            label_encoder,
            "Default_Model_Diffusion_ttopa",
            features,
            master_df,
            step=f"TTOPA_Step_{step}",
            attack="Diffusion"
        )
        save_confusion_matrix_and_data(y_true, y_pred, label_encoder.classes_, f"results/confusion_matrix_Diffusion_TTOPA_Step_{step}.png", f"results/confusion_matrix_Diffusion_TTOPA_Step_{step}.csv")

    # Save evaluation results
    master_df.to_csv(os.path.join(RESULTS_DIR, "f1_degradation_ttopa_recovery.csv"), index=False)
    print("\nF1 degradation and TTOPA recovery results saved to 'f1_degradation_ttopa_recovery.csv'.")

if __name__ == "__main__":
    main()
