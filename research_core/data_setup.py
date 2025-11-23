import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler



class FraudDataset(Dataset):
    
    def __init__(self, X, y):
        """
        Args:
            X (DataFrame or np.array): Input features.
            y (Series or np.array): Target labels (0 or 1).
        """
       
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_process_data(file_path, batch_size=64, test_size=0.2, val_size=0.2, random_state=42):
  
    print(f"Loading data from {file_path}...")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the path.")


    scaler = RobustScaler()
    
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))


    X = df.drop('Class', axis=1)
    y = df['Class']

    total_test_val_size = val_size + test_size
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=total_test_val_size, 
        random_state=random_state, 
        stratify=y
    )


    relative_test_size = test_size / total_test_val_size
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=relative_test_size, 
        random_state=random_state, 
        stratify=y_temp
    )


    print("-" * 40)
    print(f"Train Set Shape : {X_train.shape} | Fraud Ratio: {y_train.mean():.6f}")
    print(f"Val Set Shape   : {X_val.shape}   | Fraud Ratio: {y_val.mean():.6f}")
    print(f"Test Set Shape  : {X_test.shape}  | Fraud Ratio: {y_test.mean():.6f}")
    print("-" * 40)


    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)
    test_dataset = FraudDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
  
    try:
        tr, val, ts = load_and_process_data('../data/creditcard.csv')
        print("Data setup completed successfully.")
    except Exception as e:
        print(f"Error: {e}")