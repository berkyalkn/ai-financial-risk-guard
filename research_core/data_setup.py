import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder

class FraudDataset(Dataset):
    def __init__(self, X, y):

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            

        self.X = torch.tensor(X, dtype=torch.float32)
        
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        if len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(1)
            
        self.y = y_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_loaders(X, y, batch_size, val_size, test_size):

    if isinstance(y, pd.Series):
        y = y.values

    total_test_val = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=total_test_val, random_state=42, stratify=y
    )
    
    relative_test = test_size / total_test_val
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=42, stratify=y_temp
    )

    print("-" * 40)
    print(f"Train Shape: {X_train.shape} | Positive Ratio: {y_train.mean():.4f}")
    print(f"Val Shape  : {X_val.shape}  | Positive Ratio: {y_val.mean():.4f}")
    print(f"Test Shape : {X_test.shape}  | Positive Ratio: {y_test.mean():.4f}")
    print("-" * 40)

    train_dl = DataLoader(FraudDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(FraudDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(FraudDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    
    return train_dl, val_dl, test_dl, input_dim


def load_creditcard_data(file_path, batch_size=64):
    print(f"\nLoading Credit Card Data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        alt_path = file_path.replace("../", "")
        print(f"Path not found. Trying: {alt_path}")
        df = pd.read_csv(alt_path)

    scaler = RobustScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    X = df.drop('Class', axis=1)
    y = df['Class']

    return create_loaders(X, y, batch_size, 0.2, 0.2)


def load_churn_data(file_path, batch_size=64):
    print(f"\nLoading Churn Data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        alt_path = file_path.replace("../", "")
        print(f"Path not found. Trying: {alt_path}")
        df = pd.read_csv(alt_path)


    drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    le = LabelEncoder()

    if 'Gender' in df.columns:
        df['Gender'] = le.fit_transform(df['Gender'])
    
    if 'Geography' in df.columns:
        df = pd.get_dummies(df, columns=['Geography'], drop_first=True, dtype=int)
    
    cols_to_scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    existing_cols = [c for c in cols_to_scale if c in df.columns]
    
    scaler = StandardScaler()
    df[existing_cols] = scaler.fit_transform(df[existing_cols])

    X = df.drop('Exited', axis=1)
    y = df['Exited']

   
    X = X.astype(float) 

    return create_loaders(X, y, batch_size, 0.2, 0.2)

if __name__ == "__main__":

    try:
        _, _, _, dim1 = load_creditcard_data('../data/creditcard.csv')
        print(f"Credit Card Loaded. Input Dim: {dim1}")
    except Exception as e:
        print(f"Credit Card Error: {e}")


    try:
        _, _, _, dim2 = load_churn_data('../data/Churn_Modelling.csv')
        print(f"Churn Data Loaded. Input Dim: {dim2}")
    except Exception as e:
        print(f"Churn Data Error: {e}")
        import traceback
        traceback.print_exc()