import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os


from data_setup import load_creditcard_data, load_churn_data
from models import ShallowMLP, DeepFraudMLP, ResNetMLP
from losses import WeightedBCELoss, FocalLoss, PolyLoss, DiceLoss
from train import train_model
from visualization import plot_results


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_experiment_configs(input_dim, device):
    experiments = {
     
        "EXP_01_Shallow_BCE": {
            "model": ShallowMLP(input_dim=input_dim),
            "criterion": nn.BCEWithLogitsLoss(),
            "description": "Shallow Network with Standard BCE"
        },

        "EXP_02_Shallow_WeightedBCE": {
            "model": ShallowMLP(input_dim=input_dim),
            "criterion": WeightedBCELoss(pos_weight=5.0, device=device),
            "description": "Shallow Network with Weighted BCE"
        },

        "EXP_03_Shallow_Focal": {
            "model": ShallowMLP(input_dim=input_dim),
            "criterion": FocalLoss(gamma=2.0, alpha=0.25),
            "description": "Shallow Network with Focal Loss"
        },

        "EXP_04_Shallow_Poly": {
            "model": ShallowMLP(input_dim=input_dim),
            "criterion": PolyLoss(epsilon=1.0),
            "description": "Shallow Network with PolyLoss"
        },

        "EXP_05_Shallow_Dice": {
            "model": ShallowMLP(input_dim=input_dim),
            "criterion": DiceLoss(smooth=1.0),
            "description": "Shallow Network with Dice Loss"
        },

        "EXP_06_Deep_BCE": {
            "model": DeepFraudMLP(input_dim=input_dim),
            "criterion": nn.BCEWithLogitsLoss(),
            "description": "Deep Network with Standard BCE"
        },

        "EXP_07_Deep_WeightedBCE": {
            "model": DeepFraudMLP(input_dim=input_dim),
            "criterion": WeightedBCELoss(pos_weight=5.0, device=device), 
            "description": "Deep Network with Weighted BCE"
        },

        "EXP_08_Deep_Focal": {
            "model": DeepFraudMLP(input_dim=input_dim),
            "criterion": FocalLoss(gamma=2.0, alpha=0.25),
            "description": "Deep Network with Focal Loss (Gamma=2.0)"
        },

        "EXP_09_Deep_Poly": {
            "model": DeepFraudMLP(input_dim=input_dim),
            "criterion": PolyLoss(epsilon=1.0),
            "description": "Deep Network with PolyLoss"
        },

        "EXP_10_Deep_Dice": {
            "model": DeepFraudMLP(input_dim=input_dim),
            "criterion": DiceLoss(smooth=1.0),
            "description": "Deep Network with Dice Loss"
        },

        "EXP_11_ResNet_BCE": {
            "model": ResNetMLP(input_dim=input_dim),
            "criterion": nn.BCEWithLogitsLoss(),
            "description": "Residual Network with Standard BCE"
        },

        "EXP_12_ResNet_WeightedBCE": {
            "model": ResNetMLP(input_dim=input_dim),
            "criterion": WeightedBCELoss(pos_weight=5.0, device=device),
            "description": "Residual Network with Weighted BCE" 
        },

        "EXP_13_ResNet_Focal": {
            "model": ResNetMLP(input_dim=input_dim),
            "criterion": FocalLoss(gamma=2.0, alpha=0.25),
            "description": "Residual Network with Focal Loss"
        },

        "EXP_14_ResNet_Poly": {
            "model": ResNetMLP(input_dim=input_dim),
            "criterion": PolyLoss(epsilon=1.0),
            "description": "Residual Network with PolyLoss"
        },

        "EXP_15_ResNet_Dice": {
            "model": ResNetMLP(input_dim=input_dim),
            "criterion": DiceLoss(smooth=1.0),
            "description": "Residual Network with Dice Loss"
        },

        "EXP_16_ResNet_Focal_G05": {
            "model": ResNetMLP(input_dim=input_dim),
            "criterion": FocalLoss(gamma=0.5, alpha=0.25),
            "description": "Ablation: ResNet with Low Gamma (0.5)"
        },

        "EXP_17_ResNet_Focal_G50": {
            "model": ResNetMLP(input_dim=input_dim),
            "criterion": FocalLoss(gamma=5.0, alpha=0.25),
            "description": "Ablation: ResNet with High Gamma (5.0)"

        },

        "EXP_18_Weighted_W1": { 
            "model": ResNetMLP(input_dim=input_dim),
            "criterion": WeightedBCELoss(pos_weight=1.0, device=device),
            "description": "Ablation: Weighted BCE with Neutral Weight (1.0)"
        },
        "EXP_19_Weighted_W100": { 
            "model": ResNetMLP(input_dim=input_dim),
            "criterion": WeightedBCELoss(pos_weight=100.0, device=device),
            "description": "Ablation: Weighted BCE with High Weight (100.0)"
        }
    }
    return experiments

def run_dataset_suite(dataset_name, loader_func, file_path, device, epochs=10):
    print(f"\n{'#'*60}")
    print(f"DATASET STARTING: {dataset_name}")
    print(f"{'#'*60}")

    try:
        train_loader, val_loader, test_loader, input_dim = loader_func(file_path, batch_size=1024)
    except Exception as e:
        print(f"Skipping {dataset_name} due to loading error: {e}")
        return

    experiments = get_experiment_configs(input_dim, device)
    
    results = {}
    histories = {}

    total_exps = len(experiments)
    current_exp = 1
    
    for exp_name, config in experiments.items():
        print(f"\n--- [{current_exp}/{total_exps}] Running {exp_name} on {dataset_name} ---")
        print(f"{config['description']}")
        
        model = config['model'].to(device)
        criterion = config['criterion']
        optimizer = optim.Adam(model.parameters(), lr=0.001) 
        
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=epochs, device=device
        )
        
        best_score = max(history['val_prauc'])
        results[exp_name] = best_score
        histories[exp_name] = history
        print(f"Finished. Best PR-AUC: {best_score:.4f}")
        current_exp += 1

    save_folder = f"results/{dataset_name}"
    plot_results(histories, save_dir=save_folder)

    print(f"\nLEADERBOARD FOR {dataset_name} (PR-AUC)")
    print("-" * 60)

    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for rank, (name, score) in enumerate(sorted_res, 1):
        print(f"{rank}. {name:<30} : {score:.4f}")
    print("-" * 60)


def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main Experiment started on device: {device}")

    run_dataset_suite(
        dataset_name="CreditCard",
        loader_func=load_creditcard_data,
        file_path="../data/creditcard.csv",
        device=device,
        epochs=10
    )

    run_dataset_suite(
        dataset_name="Churn",
        loader_func=load_churn_data,
        file_path="../data/Churn_Modelling.csv",
        device=device,
        epochs=15
    )

    print("\n ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()