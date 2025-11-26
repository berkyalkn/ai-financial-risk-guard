import torch
import torch.optim as optim
import os
import sys

from data_setup import load_creditcard_data
from models import ResNetMLP
from losses import PolyLoss
from train import train_model

def save_best():
    print("Best Model (ResNet + PolyLoss) is being prepared...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    BATCH_SIZE = 1024
    EPOCHS = 10 
    LR = 0.001
    
    data_path = '../data/creditcard.csv'
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found: {data_path}")
        return

    train_loader, val_loader, _, input_dim = load_creditcard_data(data_path, batch_size=BATCH_SIZE)
    print(f"Data loaded. Input Size: {input_dim}")

    model = ResNetMLP(input_dim=input_dim).to(device)
    
    criterion = PolyLoss(epsilon=1.0)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Training begins...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=EPOCHS, device=device
    )
    
    save_dir = "../saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = f"{save_dir}/fraud_resnet_poly.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nThe model has been successfully saved: {save_path}")

if __name__ == "__main__":
    save_best()