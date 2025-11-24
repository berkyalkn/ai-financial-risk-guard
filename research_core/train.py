import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
import copy

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calculates F1, Precision, Recall, and PR-AUC.
    Args:
        y_true: Ground truth labels.
        y_pred_probs: Predicted probabilities.
        threshold: Decision threshold.
    """

    if np.isnan(y_pred_probs).any():
        y_pred_probs = np.nan_to_num(y_pred_probs, nan=0.0)

    y_pred = (y_pred_probs >= threshold).astype(int)
    
    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'pr_auc': average_precision_score(y_true, y_pred_probs)
    }
    return metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train() 
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() 

        outputs = model(inputs) 

        loss = criterion(outputs, labels)

        loss.backward() 

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step() 

        running_loss += loss.item() * inputs.size(0)
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()

        if np.isnan(probs).any():
            probs = np.nan_to_num(probs, nan=0.0)


        all_preds.append(probs)
        all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    metrics = calculate_metrics(all_targets, all_preds)
    return epoch_loss, metrics


def evaluate(model, dataloader, criterion, device):

    model.eval() 
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad(): 
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs).detach().cpu().numpy()

            if np.isnan(probs).any():
                probs = np.nan_to_num(probs, nan=0.0)


            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    metrics = calculate_metrics(all_targets, all_preds)
    return epoch_loss, metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cpu'):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_prauc = 0.0 
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_prauc': [], 'val_prauc': []
    }

    
    for epoch in range(num_epochs):

        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_prauc'].append(train_metrics['pr_auc'])
        history['val_prauc'].append(val_metrics['pr_auc'])

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} F1: {train_metrics['f1']:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_metrics['f1']:.4f} PR-AUC: {val_metrics['pr_auc']:.4f}")

        if val_metrics['pr_auc'] > best_prauc:
            best_prauc = val_metrics['pr_auc']
            best_model_wts = copy.deepcopy(model.state_dict())
            print("  --> New Best Model Saved!")

        if not np.isnan(val_metrics['pr_auc']) and val_metrics['pr_auc'] > best_prauc:
            best_prauc = val_metrics['pr_auc']
            best_model_wts = copy.deepcopy(model.state_dict())


    model.load_state_dict(best_model_wts)
    return model, history

