import torch
import torch.nn as nn

class ShallowMLP(nn.Module):
    """
    Baseline Model: A shallow neural network with only one hidden layer.
    
    Purpose: To demonstrate that simple architectures struggle with 
    complex/imbalanced patterns compared to deep architectures.
    
    Architecture:
        Input (30 features) -> Hidden (64 neurons) -> ReLU -> Output (1 neuron)
    """
    def __init__(self, input_dim=30, hidden_dim=64):
        super(ShallowMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.output_layer(x)
       
        return x


class DeepFraudMLP(nn.Module):
    """
    Main Model: A Deep Feed-Forward Neural Network.

    Architecture:
        Input -> [Linear -> BN -> ReLU -> Dropout] x N -> Output
    """
    def __init__(self, input_dim=30, hidden_dims=[256, 128, 64], dropout_rate=0.3):
     
        super(DeepFraudMLP, self).__init__()
        
        layers = []
        
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))         
            layers.append(nn.BatchNorm1d(h_dim))             
            layers.append(nn.ReLU())                         
            layers.append(nn.Dropout(p=dropout_rate))        
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """
    Building Block for ResNet: Skip Connection Block
    It adds the input (x) to the processed output (f(x)): Output = x + f(x)
    """
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
    """
    The difference from a standard MLP is the presence of information highways (ResBlocks) between layers.
    This allows training of deeper networks.
    """
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