import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
  
    def __init__(self, pos_weight=1.0, device='cpu'):
     
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight]).to(device)

    def forward(self, inputs, targets):
      
        return F.binary_cross_entropy_with_logits(
            inputs, 
            targets, 
            pos_weight=self.pos_weight
        )

class FocalLoss(nn.Module):
 
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
     
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
       
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss) 
        
        focal_term = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_term = alpha_t * focal_term
            
        loss = focal_term * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class PolyLoss(nn.Module):
    
    def __init__(self, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        loss = ce_loss + self.epsilon * (1 - pt)
        return loss.mean()