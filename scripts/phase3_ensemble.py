#!/usr/bin/env python3
"""
PHASE 3: ENSEMBLE OF CNN + LSTM (RESEARCH-GRADE)
===================================================

Objective: Combine DS-1D-CNN and LSTM via weighted ensemble to maximize F1.

Key research principles:
  1. Complementary models: CNN (spatial) + LSTM (temporal)
  2. Probabilistic fusion: Use softmax probabilities, not hard predictions
  3. Weight optimization: Grid search on validation set
  4. Calibration: Temperature scaling per model
  5. Stacking (optional): Train meta-learner on ensemble outputs
  
Expected performance: 90.0-91.0% F1
  - CNN: 88-89% (improved with Phase 1)
  - LSTM: 89-89.8%
  - Ensemble: 90-91% (due to complementary errors)

Ensemble variants:
  1. Simple weighted averaging (fastest, robust)
  2. Stacking with meta-learner (potentially +0.2-0.5%)
  3. Voting with confidence thresholds
  
We'll implement all three and compare.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from scipy.optimize import minimize
import json

# ============================================================================
# ENSEMBLE METHODS
# ============================================================================

class WeightedEnsemble:
    """
    Weighted averaging ensemble: p_ensemble = w1*p_cnn + w2*p_lstm
    
    Simplest and most robust. Proven to work well on diverse models.
    """
    
    def __init__(self, cnn_model: nn.Module, lstm_model: nn.Module,
                 w_cnn: float = 0.5, w_lstm: float = 0.5,
                 temp_cnn: float = 1.0, temp_lstm: float = 1.0):
        """
        Args:
            cnn_model: Trained CNN model
            lstm_model: Trained LSTM model
            w_cnn: Weight for CNN predictions (0-1)
            w_lstm: Weight for LSTM predictions (0-1)
            temp_cnn: Temperature scaling for CNN
            temp_lstm: Temperature scaling for LSTM
        """
        self.cnn_model = cnn_model
        self.lstm_model = lstm_model
        self.w_cnn = w_cnn
        self.w_lstm = w_lstm
        self.temp_cnn = temp_cnn
        self.temp_lstm = temp_lstm
        
        # Normalize weights
        total = w_cnn + w_lstm
        self.w_cnn /= total
        self.w_lstm /= total
    
    def predict_proba(self, x: torch.Tensor, device='cuda') -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            x: Input batch (batch, features)
            device: cuda/cpu
        
        Returns:
            probs: (batch, n_classes) ensemble probabilities
        """
        self.cnn_model.eval()
        self.lstm_model.eval()
        
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=device)
            else:
                x = x.to(device)
            
            # CNN predictions
            cnn_logits = self.cnn_model(x)
            cnn_probs = F.softmax(cnn_logits / self.temp_cnn, dim=1)
            
            # LSTM predictions
            lstm_logits = self.lstm_model(x)
            lstm_probs = F.softmax(lstm_logits / self.temp_lstm, dim=1)
            
            # Weighted ensemble
            ensemble_probs = (self.w_cnn * cnn_probs + 
                            self.w_lstm * lstm_probs)
        
        return ensemble_probs.cpu().numpy()
    
    def predict(self, x: torch.Tensor, device='cuda') -> np.ndarray:
        """Hard predictions (argmax)"""
        probs = self.predict_proba(x, device=device)
        return np.argmax(probs, axis=1)

class StackingEnsemble(nn.Module):
    """
    Stacking ensemble: Train meta-learner on CNN+LSTM predictions.
    
    Procedure:
      1. Get CNN and LSTM outputs on validation set (features for meta-learner)
      2. Train simple meta-learner (logistic regression or small MLP)
      3. Use meta-learner for final predictions
      
    Advantages: Can learn non-linear combination of models
    Disadvantage: More complex, risk of overfitting on small validation set
    """
    
    def __init__(self, n_classes: int = 18, hidden_dim: int = 64):
        super().__init__()
        
        # Meta-learner: MLP that takes concatenated outputs of CNN+LSTM
        # Input: 2*n_classes (CNN logits + LSTM logits)
        # Output: n_classes
        self.meta = nn.Sequential(
            nn.Linear(2 * n_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, cnn_logits: torch.Tensor, lstm_logits: torch.Tensor):
        """
        Args:
            cnn_logits: (batch, n_classes) from CNN
            lstm_logits: (batch, n_classes) from LSTM
        
        Returns:
            ensemble_logits: (batch, n_classes)
        """
        # Concatenate CNN and LSTM logits
        combined = torch.cat([cnn_logits, lstm_logits], dim=1)
        ensemble_logits = self.meta(combined)
        return ensemble_logits

# ============================================================================
# WEIGHT OPTIMIZATION
# ============================================================================

def optimize_ensemble_weights(cnn_model: nn.Module, lstm_model: nn.Module,
                             X_val: torch.Tensor, y_val: np.ndarray,
                             device: str = 'cuda') -> tuple:
    """
    Find optimal weights for weighted ensemble using grid search.
    
    Maximizes: Macro-F1 on validation set
    
    Args:
        cnn_model: Trained CNN model
        lstm_model: Trained LSTM model
        X_val: Validation features
        y_val: Validation labels
        device: cuda/cpu
    
    Returns:
        best_w: (w_cnn, w_lstm) optimal normalized weights
        best_f1: F1 at optimal weights
    """
    
    print("\n[Ensemble] Optimizing weights via grid search...")
    
    cnn_model.eval()
    lstm_model.eval()
    
    # Get model outputs
    with torch.no_grad():
        if not isinstance(X_val, torch.Tensor):
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        else:
            X_val_t = X_val.to(device)
        
        cnn_probs = F.softmax(cnn_model(X_val_t), dim=1).cpu().numpy()
        lstm_probs = F.softmax(lstm_model(X_val_t), dim=1).cpu().numpy()
    
    if isinstance(y_val, torch.Tensor):
        y_val = y_val.cpu().numpy()
    
    # Grid search: w_cnn from 0.1 to 0.9 in steps of 0.1
    best_f1 = 0.0
    best_w = (0.5, 0.5)
    
    print("  Weight        F1-Score    Improvement")
    print("  " + "-"*40)
    
    for w_cnn_raw in np.arange(0.1, 1.0, 0.1):
        w_lstm_raw = 1.0 - w_cnn_raw
        
        # Normalize
        total = w_cnn_raw + w_lstm_raw
        w_cnn = w_cnn_raw / total
        w_lstm = w_lstm_raw / total
        
        # Ensemble predictions
        ensemble_probs = w_cnn * cnn_probs + w_lstm * lstm_probs
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        f1 = f1_score(y_val, ensemble_preds, average='macro', zero_division=0)
        
        improvement = f1 - f1_score(y_val, np.argmax(cnn_probs, axis=1), average='macro', zero_division=0)
        
        marker = " ✓" if f1 > best_f1 else ""
        print(f"  CNN={w_cnn:.2f} LSTM={w_lstm:.2f}   {f1:.4f}    +{improvement:.4f}{marker}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_w = (w_cnn, w_lstm)
    
    print(f"\n  ✓ Optimal weights: CNN={best_w[0]:.2f}, LSTM={best_w[1]:.2f}")
    print(f"  ✓ Best F1: {best_f1:.4f}")
    
    return best_w, best_f1

def find_optimal_temperature_ensemble(cnn_model: nn.Module, lstm_model: nn.Module,
                                     X_val: torch.Tensor, y_val: np.ndarray,
                                     w_cnn: float = 0.5, w_lstm: float = 0.5,
                                     device: str = 'cuda') -> tuple:
    """
    Find optimal temperatures for CNN and LSTM separately, then for ensemble.
    
    Args:
        cnn_model: Trained CNN model
        lstm_model: Trained LSTM model
        X_val: Validation features
        y_val: Validation labels
        w_cnn, w_lstm: Ensemble weights
        device: cuda/cpu
    
    Returns:
        (temp_cnn, temp_lstm), best_f1
    """
    
    print("\n[Ensemble] Optimizing temperatures...")
    
    cnn_model.eval()
    lstm_model.eval()
    
    with torch.no_grad():
        if not isinstance(X_val, torch.Tensor):
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        else:
            X_val_t = X_val.to(device)
        
        cnn_logits = cnn_model(X_val_t).cpu().numpy()
        lstm_logits = lstm_model(X_val_t).cpu().numpy()
    
    if isinstance(y_val, torch.Tensor):
        y_val = y_val.cpu().numpy()
    
    best_f1_overall = 0.0
    best_temps = (1.0, 1.0)
    
    for temp_cnn in np.arange(0.7, 1.5, 0.1):
        for temp_lstm in np.arange(0.7, 1.5, 0.1):
            # Scale logits
            cnn_probs = torch.softmax(torch.tensor(cnn_logits / temp_cnn), dim=1).numpy()
            lstm_probs = torch.softmax(torch.tensor(lstm_logits / temp_lstm), dim=1).numpy()
            
            # Ensemble
            ensemble_probs = w_cnn * cnn_probs + w_lstm * lstm_probs
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            
            f1 = f1_score(y_val, ensemble_preds, average='macro', zero_division=0)
            
            if f1 > best_f1_overall:
                best_f1_overall = f1
                best_temps = (temp_cnn, temp_lstm)
    
    print(f"  ✓ Optimal temperatures: CNN={best_temps[0]:.2f}, LSTM={best_temps[1]:.2f}")
    print(f"  ✓ Ensemble F1: {best_f1_overall:.4f}")
    
    return best_temps, best_f1_overall

# ============================================================================
# ENSEMBLE EVALUATION
# ============================================================================

def evaluate_ensemble(ensemble, X_val: torch.Tensor, y_val: np.ndarray,
                     cnn_model: nn.Module, lstm_model: nn.Module,
                     device: str = 'cuda') -> dict:
    """
    Comprehensive ensemble evaluation.
    
    Returns:
      - Per-model performance (CNN, LSTM)
      - Ensemble performance
      - Metrics (Macro-F1, Accuracy, etc.)
      - Confusion matrix
    """
    
    if isinstance(y_val, torch.Tensor):
        y_val = y_val.cpu().numpy()
    
    # Individual model predictions
    cnn_model.eval()
    lstm_model.eval()
    
    with torch.no_grad():
        if not isinstance(X_val, torch.Tensor):
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        else:
            X_val_t = X_val.to(device)
        
        cnn_preds = torch.argmax(cnn_model(X_val_t), dim=1).cpu().numpy()
        lstm_preds = torch.argmax(lstm_model(X_val_t), dim=1).cpu().numpy()
    
    # Ensemble predictions
    ensemble_preds = ensemble.predict(X_val, device=device)
    
    # Metrics
    cnn_f1 = f1_score(y_val, cnn_preds, average='macro', zero_division=0)
    cnn_acc = accuracy_score(y_val, cnn_preds)
    
    lstm_f1 = f1_score(y_val, lstm_preds, average='macro', zero_division=0)
    lstm_acc = accuracy_score(y_val, lstm_preds)
    
    ensemble_f1 = f1_score(y_val, ensemble_preds, average='macro', zero_division=0)
    ensemble_acc = accuracy_score(y_val, ensemble_preds)
    
    results = {
        'cnn': {'f1': cnn_f1, 'acc': cnn_acc},
        'lstm': {'f1': lstm_f1, 'acc': lstm_acc},
        'ensemble': {'f1': ensemble_f1, 'acc': ensemble_acc},
        'improvement_over_cnn': ensemble_f1 - cnn_f1,
        'improvement_over_lstm': ensemble_f1 - lstm_f1,
        'ensemble_preds': ensemble_preds.tolist()
    }
    
    return results

# ============================================================================
# SAVE ENSEMBLE
# ============================================================================

def save_ensemble(ensemble: WeightedEnsemble, results: dict, ckpt_dir: Path):
    """Save ensemble weights and results"""
    
    torch.save({
        'w_cnn': ensemble.w_cnn,
        'w_lstm': ensemble.w_lstm,
        'temp_cnn': ensemble.temp_cnn,
        'temp_lstm': ensemble.temp_lstm,
        'results': results
    }, ckpt_dir / 'phase3_ensemble_weights.pt')
    
    with open(ckpt_dir / 'phase3_ensemble_results.json', 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'ensemble_preds'}, f, indent=2)
    
    print(f"✓ Saved ensemble: {ckpt_dir / 'phase3_ensemble_weights.pt'}")

if __name__ == '__main__':
    print("Ensemble utilities ready. Use in train_phase3_ensemble.py")
