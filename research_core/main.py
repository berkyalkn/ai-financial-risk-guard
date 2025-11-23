import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os


from data_setup import load_and_process_data as load_data
from models import ShallowMLP, DeepFraudMLP
from losses import WeightedBCELoss, FocalLoss, PolyLoss
from train import train_model


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_experiments(device):
    experiments = {
     
        "EXP_01_Shallow_BCE": {
            "model": ShallowMLP(input_dim=30),
            "criterion": nn.BCEWithLogitsLoss(),
            "description": "Baseline: Shallow Network with Standard BCE"
        }, 

        "EXP_02_Shallow_WeightedBCE": {
            "model": ShallowMLP(input_dim=30),
            "criterion": WeightedBCELoss(pos_weight=5.0, device=device),
            "description": "Shallow Network with Weighted BCE" 
        }, 

        "EXP_03_Shallow_Focal": {
            "model": ShallowMLP(input_dim=30),
            "criterion": FocalLoss(gamma=2.0, alpha=0.25),
            "description": "Shallow Network with Focal Loss"
        }, 
        
        "EXP_04_Deep_BCE": {
            "model": DeepFraudMLP(input_dim=30),
            "criterion": nn.BCEWithLogitsLoss(),
            "description": "Benchmark: Deep Network with Standard BCE"
        },
        
        "EXP_05_Deep_WeightedBCE": {
            "model": DeepFraudMLP(input_dim=30),
            "criterion": WeightedBCELoss(pos_weight=5.0, device=device), 
            "description": "Deep Network with Weighted BCE"
        },
        
        "EXP_06_Deep_Focal": {
            "model": DeepFraudMLP(input_dim=30),
            "criterion": FocalLoss(gamma=2.0, alpha=0.25),
            "description": "Deep Network with Focal Loss (Gamma=2.0)"
        },

        "EXP_07_Deep_Poly": {
            "model": DeepFraudMLP(input_dim=30),
            "criterion": PolyLoss(epsilon=1.0),
            "description": "Experimental: Deep Network with PolyLoss (Epsilon=1.0)"
        }
    }
    return experiments


def run_experiments():
    SEED = 42
    BATCH_SIZE = 1024 
    EPOCHS = 10    
    LR = 0.001       
    DATA_PATH = '../data/creditcard.csv'
    
    set_seed(SEED)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiments on: {device}")


    print("\n[1/3] Loading Data...")
    try:
        train_loader, val_loader, test_loader = load_data(DATA_PATH, batch_size=BATCH_SIZE)
    except FileNotFoundError:
    
        print("Data path error.")
        train_loader, val_loader, test_loader = load_data('data/creditcard.csv', batch_size=BATCH_SIZE)


    print("\n[2/3] Starting Experiments...")
    experiments = get_experiments(device)
    results = {}

    for exp_name, config in experiments.items():
        print(f"\n{'='*60}")
        print(f"Running: {exp_name}")
        print(f"Info: {config['description']}")
        print(f"{'='*60}")
        
    
        model = config['model'].to(device)
        criterion = config['criterion']
        
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        trained_model, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            num_epochs=EPOCHS, 
            device=device
        )
        
        best_score = max(history['val_prauc'])
        results[exp_name] = best_score
        print(f"🏆 {exp_name} Finished. Best Val PR-AUC: {best_score:.4f}")

    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY (Val PR-AUC)")
    print(f"{'='*60}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (name, score) in enumerate(sorted_results, 1):
        print(f"{rank}. {name:<25} : {score:.4f}")
        
    print(f"{'='*60}")
    print("All experiments completed.")


if __name__ == "__main__":
    run_experiments()