import torch
import torch.nn as nn
import os
import numpy as np


class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout_rate):

        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)

class ResNetMLP(nn.Module):

    def __init__(self, input_dim=30, hidden_dim=128, num_blocks=3, dropout_rate=0.3):

        super(ResNetMLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)


class FraudDetectionService:
    def __init__(self):
        self.device = torch.device("cpu") 
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
       
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../../saved_models/fraud_resnet_poly.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Model is loading: {model_path}")
        
        model = ResNetMLP(input_dim=30, hidden_dim=128, num_blocks=3, dropout_rate=0.3)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        
        print("The model is successfully loaded and ready.")
        return model

    def predict(self, features: list):
      
        input_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()
            
        is_fraud = probability > 0.5
        
        if probability > 0.8:
            risk_level = "CRITICAL"
        elif probability > 0.5:
            risk_level = "HIGH"
        elif probability > 0.2:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return {
            "is_fraud": is_fraud,
            "fraud_probability": probability,
            "risk_level": risk_level
        }

fraud_service = FraudDetectionService()