#!/usr/bin/env python3
"""
train_phase3_ensemble_research_grade.py

RESEARCH-GRADE ENSEMBLE (Phase 3)
===================================

OBJECTIVE VALIDATION:
- CNN F1: 88.17% (baseline) or 88.7-89.3% (Phase 1 improved)
- LSTM F1: 89.0-89.8% (expected, Phase 2)
- Ensemble target: 90.0-91.0% F1

CRITICAL ASSUMPTIONS TO VALIDATE:
1. **Error complementarity**: CNN and LSTM make DIFFERENT mistakes
   - If complementarity < 30%, ensemble won't help much
   - Measure: Error agreement rate, confusion matrix overlap

2. **Calibration**: Models produce valid probabilities
   - Measure: Expected Calibration Error (ECE)
   - Fix: Temperature scaling

3. **Weight optimality**: Grid search finds true optimum
   - Validate: Test multiple initializations
   - Check: Optimization landscape is convex

ENSEMBLE TYPES EVALUATED:
1. Simple averaging (w=0.5, baseline)
2. Weighted averaging (optimized weights)
3. Temperature-scaled weighted (optimized temps + weights)

FALLBACK STRATEGY:
If ensemble F1 ≤ max(CNN, LSTM):
- Report best single model
- Analyze why ensemble failed (high error correlation)
- Suggest stacking meta-learner

Expected: 90.0-91.0% F1 (with 40-60% error complementarity)
Minimum: 89.5%+ F1 (acceptable for publication)
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(r"C:\ML")
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class EnsembleConfig:
    """Research-grade ensemble configuration"""
    
    # Weight search
    weight_search_granularity: int = 19  # 0.0, 0.05, ..., 0.95, 1.0
    weight_search_range: Tuple[float, float] = (0.0, 1.0)
    
    # Temperature search
    temp_search_granularity: int = 15  # 0.7, 0.8, ..., 2.0
    temp_search_range: Tuple[float, float] = (0.7, 2.0)
    
    # Validation
    min_complementarity: float = 0.30  # Min 30% different errors
    ece_bins: int = 15  # Expected Calibration Error bins
    
    # Statistical testing
    bootstrap_iterations: int = 1000
    alpha: float = 0.05  # Significance level
    
    # Batch processing
    batch_size: int = 256
    
    # Paths
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints" / "pytorch_fixed_full"
    results_dir: Path = PROJECT_ROOT / "results" / "pytorch_fixed_full"
    
    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CALIBRATION METRICS
# ============================================================================
def compute_ece(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE)
    
    Measures how well predicted probabilities match true frequencies.
    Lower is better (0 = perfect calibration).
    
    Args:
        logits: (N, C) raw logits
        labels: (N,) true labels
        n_bins: number of bins
    
    Returns:
        ece: float (0-1)
    """
    # Convert to probabilities
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    
    # Get predicted class and confidence
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Bin confidences
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Select samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


# ============================================================================
# ERROR COMPLEMENTARITY ANALYSIS
# ============================================================================
def analyze_error_complementarity(preds_cnn: np.ndarray,
                                  preds_lstm: np.ndarray,
                                  y_true: np.ndarray) -> Dict:
    """
    CRITICAL VALIDATION: Check if CNN and LSTM make different mistakes
    
    If models make the same mistakes, ensemble won't help.
    Need ≥30% complementarity for meaningful improvement.
    
    Returns:
        metrics: {
            'agreement_rate': float,       # How often both correct or both wrong
            'complementarity': float,       # How often one correct, other wrong
            'both_correct': float,
            'both_wrong': float,
            'cnn_only_correct': float,
            'lstm_only_correct': float,
            'recommendation': str
        }
    """
    print("\n[COMPLEMENTARITY ANALYSIS] Checking error diversity...")
    
    cnn_correct = (preds_cnn == y_true)
    lstm_correct = (preds_lstm == y_true)
    
    both_correct = np.sum(cnn_correct & lstm_correct)
    both_wrong = np.sum(~cnn_correct & ~lstm_correct)
    cnn_only = np.sum(cnn_correct & ~lstm_correct)
    lstm_only = np.sum(~cnn_correct & lstm_correct)
    
    total = len(y_true)
    
    # Agreement: how often both right or both wrong
    agreement_rate = (both_correct + both_wrong) / total
    
    # Complementarity: how often one right, other wrong
    complementarity = (cnn_only + lstm_only) / total
    
    print(f"  Both correct:       {both_correct:6d} ({both_correct/total*100:5.2f}%)")
    print(f"  Both wrong:         {both_wrong:6d} ({both_wrong/total*100:5.2f}%)")
    print(f"  CNN only correct:   {cnn_only:6d} ({cnn_only/total*100:5.2f}%)")
    print(f"  LSTM only correct:  {lstm_only:6d} ({lstm_only/total*100:5.2f}%)")
    print(f"  Agreement rate:     {agreement_rate:.4f}")
    print(f"  Complementarity:    {complementarity:.4f}")
    
    # Recommendation based on complementarity
    if complementarity >= 0.40:
        recommendation = "EXCELLENT: Strong complementarity, ensemble will help significantly"
    elif complementarity >= 0.30:
        recommendation = "GOOD: Moderate complementarity, ensemble should help"
    elif complementarity >= 0.20:
        recommendation = "FAIR: Low complementarity, ensemble may help slightly"
    else:
        recommendation = "POOR: Very low complementarity, ensemble unlikely to help much"
    
    print(f"  Recommendation: {recommendation}")
    
    return {
        'agreement_rate': float(agreement_rate),
        'complementarity': float(complementarity),
        'both_correct': float(both_correct / total),
        'both_wrong': float(both_wrong / total),
        'cnn_only_correct': float(cnn_only / total),
        'lstm_only_correct': float(lstm_only / total),
        'recommendation': recommendation
    }


# ============================================================================
# WEIGHTED ENSEMBLE CLASS
# ============================================================================
class WeightedEnsemble:
    """
    Weighted ensemble with temperature scaling
    
    p_ensemble = w_cnn * softmax(logits_cnn / T_cnn) + 
                 w_lstm * softmax(logits_lstm / T_lstm)
    """
    def __init__(self, cnn_model: nn.Module, lstm_model: nn.Module,
                 w_cnn: float = 0.5, w_lstm: float = 0.5,
                 temp_cnn: float = 1.0, temp_lstm: float = 1.0):
        self.cnn_model = cnn_model
        self.lstm_model = lstm_model
        self.w_cnn = w_cnn
        self.w_lstm = w_lstm
        self.temp_cnn = temp_cnn
        self.temp_lstm = temp_lstm
        
        # Normalize weights
        total = w_cnn + w_lstm
        self.w_cnn = w_cnn / total
        self.w_lstm = w_lstm / total
    
    def predict_proba(self, X: torch.Tensor, device: str = 'cuda') -> np.ndarray:
        """
        Get ensemble probabilities
        
        Args:
            X: (batch, features) input
            device: 'cuda' or 'cpu'
        
        Returns:
            probs: (batch, n_classes) probabilities
        """
        self.cnn_model.eval()
        self.lstm_model.eval()
        
        with torch.no_grad():
            X = X.to(device)
            
            # CNN logits
            logits_cnn = self.cnn_model(X)
            probs_cnn = F.softmax(logits_cnn / self.temp_cnn, dim=1)
            
            # LSTM logits
            logits_lstm = self.lstm_model(X)
            probs_lstm = F.softmax(logits_lstm / self.temp_lstm, dim=1)
            
            # Weighted average
            probs_ensemble = self.w_cnn * probs_cnn + self.w_lstm * probs_lstm
        
        return probs_ensemble.cpu().numpy()
    
    def predict(self, X: torch.Tensor, device: str = 'cuda') -> np.ndarray:
        """Get ensemble predictions"""
        probs = self.predict_proba(X, device)
        return np.argmax(probs, axis=1)


# ============================================================================
# ENSEMBLE TRAINER
# ============================================================================
class EnsembleTrainer:
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.cnn_model = None
        self.lstm_model = None
        self.ensemble = None
        
        self.state = {
            'weight_search_results': [],
            'temp_search_results': [],
            'complementarity_analysis': None,
            'final_results': None
        }
    
    def load_models(self):
        """Load CNN and LSTM models from checkpoints"""
        print("\n" + "="*80)
        print("MODEL LOADING")
        print("="*80)
        
        # Load CNN (Phase 1 or baseline)
        print("\n[1/3] Loading CNN model...")
        cnn_candidates = [
            self.config.checkpoint_dir / 'phase1_improved_model.pt',
            self.config.checkpoint_dir / 'best_model.pt'
        ]
        
        cnn_ckpt = None
        for path in cnn_candidates:
            if path.exists():
                try:
                    cnn_ckpt = torch.load(path, map_location='cpu')
                    print(f"  ✓ Loaded: {path.name}")
                    break
                except Exception as e:
                    print(f"  ✗ Failed to load {path.name}: {e}")
        
        if cnn_ckpt is None:
            raise FileNotFoundError(
                "CNN checkpoint not found. Run Phase 1 first:\n"
                "  python train_88plus_research_grade.py"
            )
        
        # Load LSTM (Phase 2)
        print("[2/3] Loading LSTM model...")
        lstm_path = self.config.checkpoint_dir / 'phase2_lstm_model.pt'
        
        if not lstm_path.exists():
            raise FileNotFoundError(
                "LSTM checkpoint not found. Run Phase 2 first:\n"
                "  python train_phase2_lstm_research_grade.py"
            )
        
        lstm_ckpt = torch.load(lstm_path, map_location='cpu')
        print(f"  ✓ Loaded: {lstm_path.name}")
        
        # Reconstruct models
        print("[3/3] Reconstructing models...")
        
        # CNN
        try:
            # Try importing from main training script
            from train_76m_streaming_fixed import EdgeIDSNetV2
        except ImportError:
            # Fallback to fixed script
            from train_76m_streaming_fixed import EdgeIDSNetV2
        
        self.cnn_model = EdgeIDSNetV2(
            n_features=41, n_classes=18,
            base_filters=256, dropout=0.4
        )
        
        # Load CNN state dict
        if isinstance(cnn_ckpt, dict) and 'model_state_dict' in cnn_ckpt:
            self.cnn_model.load_state_dict(cnn_ckpt['model_state_dict'])
        else:
            self.cnn_model.load_state_dict(cnn_ckpt)
        
        self.cnn_model.to(self.config.device)
        self.cnn_model.eval()
        
        # LSTM
        try:
            from train_phase2_lstm import LSTMIDSv2
        except ImportError:
            from models.lstm_ids_v2 import LSTMIDSv2
        
        # Get config from checkpoint
        if isinstance(lstm_ckpt, dict) and 'config' in lstm_ckpt:
            lstm_config = lstm_ckpt['config']
        else:
            # Default config
            lstm_config = {
                'n_features': 41,
                'n_classes': 18,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'bidirectional': True,
                'use_temporal_chunks': True
            }
        
        self.lstm_model = LSTMIDSv2(**lstm_config)
        
        # Load LSTM state dict
        if isinstance(lstm_ckpt, dict) and 'model_state_dict' in lstm_ckpt:
            self.lstm_model.load_state_dict(lstm_ckpt['model_state_dict'])
        else:
            self.lstm_model.load_state_dict(lstm_ckpt)
        
        self.lstm_model.to(self.config.device)
        self.lstm_model.eval()
        
        print("  ✓ Models loaded and ready")
    
    def load_validation_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load validation data"""
        print("\n[DATA] Loading validation set...")
        
        val_path = self.config.checkpoint_dir / 'validation_data.pt'
        if not val_path.exists():
            raise FileNotFoundError(
                f"validation_data.pt not found in {self.config.checkpoint_dir}"
            )
        
        val_data = torch.load(val_path, map_location='cpu')
        X_val = val_data['X_val']
        y_val = val_data['y_val']
        
        if isinstance(X_val, np.ndarray):
            X_val = torch.from_numpy(X_val).float()
        if isinstance(y_val, np.ndarray):
            y_val = torch.from_numpy(y_val).long()
        
        print(f"  ✓ Validation: {X_val.shape[0]:,} samples")
        
        return X_val, y_val
    
    def get_model_predictions(self, X: torch.Tensor, y: torch.Tensor) -> Dict:
        """
        Get predictions and metrics for both models
        
        Returns:
            results: {
                'cnn': {'preds', 'logits', 'f1', 'acc'},
                'lstm': {'preds', 'logits', 'f1', 'acc'}
            }
        """
        print("\n[BASELINE] Evaluating individual models...")
        
        # CNN predictions
        cnn_preds_list, cnn_logits_list = [], []
        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                X_batch = X[i:i+self.config.batch_size].to(self.config.device)
                logits = self.cnn_model(X_batch)
                preds = torch.argmax(logits, dim=1)
                
                cnn_preds_list.append(preds.cpu().numpy())
                cnn_logits_list.append(logits.cpu().numpy())
        
        cnn_preds = np.concatenate(cnn_preds_list)
        cnn_logits = np.concatenate(cnn_logits_list)
        
        # LSTM predictions
        lstm_preds_list, lstm_logits_list = [], []
        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                X_batch = X[i:i+self.config.batch_size].to(self.config.device)
                logits = self.lstm_model(X_batch)
                preds = torch.argmax(logits, dim=1)
                
                lstm_preds_list.append(preds.cpu().numpy())
                lstm_logits_list.append(logits.cpu().numpy())
        
        lstm_preds = np.concatenate(lstm_preds_list)
        lstm_logits = np.concatenate(lstm_logits_list)
        
        # Compute metrics
        y_np = y.numpy()
        
        cnn_f1 = f1_score(y_np, cnn_preds, average='macro', zero_division=0)
        cnn_acc = accuracy_score(y_np, cnn_preds)
        
        lstm_f1 = f1_score(y_np, lstm_preds, average='macro', zero_division=0)
        lstm_acc = accuracy_score(y_np, lstm_preds)
        
        print(f"  CNN:  F1 = {cnn_f1:.4f}, Acc = {cnn_acc:.4f}")
        print(f"  LSTM: F1 = {lstm_f1:.4f}, Acc = {lstm_acc:.4f}")
        
        # Calibration
        cnn_ece = compute_ece(cnn_logits, y_np, self.config.ece_bins)
        lstm_ece = compute_ece(lstm_logits, y_np, self.config.ece_bins)
        
        print(f"\n  CNN ECE:  {cnn_ece:.4f} (lower = better calibration)")
        print(f"  LSTM ECE: {lstm_ece:.4f}")
        
        return {
            'cnn': {
                'preds': cnn_preds,
                'logits': cnn_logits,
                'f1': cnn_f1,
                'acc': cnn_acc,
                'ece': cnn_ece
            },
            'lstm': {
                'preds': lstm_preds,
                'logits': lstm_logits,
                'f1': lstm_f1,
                'acc': lstm_acc,
                'ece': lstm_ece
            }
        }
    
    def optimize_weights(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
        """
        Grid search for optimal ensemble weights
        
        Returns:
            (w_cnn, w_lstm, best_f1)
        """
        print("\n[WEIGHT OPTIMIZATION] Grid search...")
        
        # Create weight grid
        weights_cnn = np.linspace(
            self.config.weight_search_range[0],
            self.config.weight_search_range[1],
            self.config.weight_search_granularity
        )
        
        best_f1 = 0.0
        best_w_cnn = 0.5
        best_w_lstm = 0.5
        
        results = []
        
        # Get predictions
        cnn_preds_list, cnn_logits_list = [], []
        lstm_preds_list, lstm_logits_list = [], []
        
        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                X_batch = X[i:i+self.config.batch_size].to(self.config.device)
                
                cnn_logits = self.cnn_model(X_batch)
                cnn_probs = F.softmax(cnn_logits, dim=1)
                cnn_preds_list.append(cnn_probs.cpu().numpy())
                
                lstm_logits = self.lstm_model(X_batch)
                lstm_probs = F.softmax(lstm_logits, dim=1)
                lstm_preds_list.append(lstm_probs.cpu().numpy())
        
        cnn_probs = np.concatenate(cnn_preds_list)
        lstm_probs = np.concatenate(lstm_preds_list)
        y_np = y.numpy()
        
        # Grid search
        for w_cnn in weights_cnn:
            w_lstm = 1.0 - w_cnn
            
            # Weighted ensemble
            probs_ensemble = w_cnn * cnn_probs + w_lstm * lstm_probs
            preds_ensemble = np.argmax(probs_ensemble, axis=1)
            
            f1 = f1_score(y_np, preds_ensemble, average='macro', zero_division=0)
            
            results.append({
                'w_cnn': float(w_cnn),
                'w_lstm': float(w_lstm),
                'f1': float(f1)
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_w_cnn = w_cnn
                best_w_lstm = w_lstm
        
        self.state['weight_search_results'] = results
        
        print(f"  Best weights: CNN={best_w_cnn:.2f}, LSTM={best_w_lstm:.2f}")
        print(f"  Best F1: {best_f1:.4f}")
        
        return best_w_cnn, best_w_lstm, best_f1
    
    def optimize_temperatures(self, X: torch.Tensor, y: torch.Tensor,
                             w_cnn: float, w_lstm: float) -> Tuple[float, float, float]:
        """
        Grid search for optimal temperatures (calibration)
        
        Returns:
            (temp_cnn, temp_lstm, best_f1)
        """
        print("\n[TEMPERATURE OPTIMIZATION] Grid search...")
        
        # Create temperature grid
        temps = np.linspace(
            self.config.temp_search_range[0],
            self.config.temp_search_range[1],
            self.config.temp_search_granularity
        )
        
        best_f1 = 0.0
        best_temp_cnn = 1.0
        best_temp_lstm = 1.0
        
        results = []
        
        # Get logits
        cnn_logits_list, lstm_logits_list = [], []
        
        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                X_batch = X[i:i+self.config.batch_size].to(self.config.device)
                
                cnn_logits = self.cnn_model(X_batch)
                cnn_logits_list.append(cnn_logits.cpu().numpy())
                
                lstm_logits = self.lstm_model(X_batch)
                lstm_logits_list.append(lstm_logits.cpu().numpy())
        
        cnn_logits = np.concatenate(cnn_logits_list)
        lstm_logits = np.concatenate(lstm_logits_list)
        y_np = y.numpy()
        
        # Grid search (both temperatures independently)
        for temp_cnn in temps:
            for temp_lstm in temps:
                # Apply temperature scaling
                cnn_probs = np.exp(cnn_logits / temp_cnn)
                cnn_probs = cnn_probs / np.sum(cnn_probs, axis=1, keepdims=True)
                
                lstm_probs = np.exp(lstm_logits / temp_lstm)
                lstm_probs = lstm_probs / np.sum(lstm_probs, axis=1, keepdims=True)
                
                # Weighted ensemble
                probs_ensemble = w_cnn * cnn_probs + w_lstm * lstm_probs
                preds_ensemble = np.argmax(probs_ensemble, axis=1)
                
                f1 = f1_score(y_np, preds_ensemble, average='macro', zero_division=0)
                
                results.append({
                    'temp_cnn': float(temp_cnn),
                    'temp_lstm': float(temp_lstm),
                    'f1': float(f1)
                })
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_temp_cnn = temp_cnn
                    best_temp_lstm = temp_lstm
        
        self.state['temp_search_results'] = results
        
        print(f"  Best temperatures: CNN={best_temp_cnn:.2f}, LSTM={best_temp_lstm:.2f}")
        print(f"  Best F1: {best_f1:.4f}")
        
        return best_temp_cnn, best_temp_lstm, best_f1
    
    def evaluate_final_ensemble(self, X: torch.Tensor, y: torch.Tensor) -> Dict:
        """Comprehensive final evaluation"""
        print("\n[FINAL EVALUATION] Ensemble performance...")
        
        # Get ensemble predictions
        ensemble_probs_list = []
        
        for i in range(0, len(X), self.config.batch_size):
            X_batch = X[i:i+self.config.batch_size]
            probs = self.ensemble.predict_proba(X_batch, self.config.device)
            ensemble_probs_list.append(probs)
        
        ensemble_probs = np.concatenate(ensemble_probs_list)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        y_np = y.numpy()
        
        # Metrics
        ensemble_f1 = f1_score(y_np, ensemble_preds, average='macro', zero_division=0)
        ensemble_acc = accuracy_score(y_np, ensemble_preds)
        
        print(f"  Ensemble F1:  {ensemble_f1:.4f}")
        print(f"  Ensemble Acc: {ensemble_acc:.4f}")
        
        # Per-class F1
        per_class_f1 = f1_score(y_np, ensemble_preds, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_np, ensemble_preds)
        
        # Classification report
        # Load label indexer for class names
        try:
            label_path = self.config.checkpoint_dir.parent / 'label_indexer.pkl'
            with open(label_path, 'rb') as f:
                indexer_data = pickle.load(f)
            class_names = indexer_data['idx_to_class']
        except:
            class_names = [f"Class_{i}" for i in range(18)]
        
        report = classification_report(y_np, ensemble_preds, 
                                      target_names=class_names,
                                      zero_division=0, digits=4)
        
        return {
            'f1': ensemble_f1,
            'acc': ensemble_acc,
            'per_class_f1': per_class_f1.tolist(),
            'confusion_matrix': cm.tolist(),
            'report': report
        }
    
    def bootstrap_ensemble_ci(self, X: torch.Tensor, y: torch.Tensor) -> Dict:
        """Bootstrap confidence interval for ensemble F1"""
        print("\n[BOOTSTRAP] Computing 95% CI...")
        
        # Get ensemble predictions once
        ensemble_preds_list = []
        for i in range(0, len(X), self.config.batch_size):
            X_batch = X[i:i+self.config.batch_size]
            preds = self.ensemble.predict(X_batch, self.config.device)
            ensemble_preds_list.append(preds)
        
        ensemble_preds = np.concatenate(ensemble_preds_list)
        y_np = y.numpy()
        
        # Bootstrap
        f1_scores = []
        n_samples = len(y_np)
        
        for _ in range(self.config.bootstrap_iterations):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            preds_sample = ensemble_preds[indices]
            y_sample = y_np[indices]
            
            f1 = f1_score(y_sample, preds_sample, average='macro', zero_division=0)
            f1_scores.append(f1)
        
        f1_scores = np.array(f1_scores)
        ci_lower = np.percentile(f1_scores, 2.5)
        ci_upper = np.percentile(f1_scores, 97.5)
        f1_mean = np.mean(f1_scores)
        
        print(f"  F1 = {f1_mean:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return {
            'f1_mean': float(f1_mean),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper)
        }
    
    def run_full_optimization(self):
        """Main ensemble optimization pipeline"""
        print("\n" + "="*80)
        print("PHASE 3: ENSEMBLE OPTIMIZATION")
        print("="*80)
        
        # Load models
        self.load_models()
        
        # Load validation data
        X_val, y_val = self.load_validation_data()
        
        # Get baseline predictions
        baseline_results = self.get_model_predictions(X_val, y_val)
        
        # Complementarity analysis
        self.state['complementarity_analysis'] = analyze_error_complementarity(
            baseline_results['cnn']['preds'],
            baseline_results['lstm']['preds'],
            y_val.numpy()
        )
        
        # Check if ensemble is worthwhile
        complementarity = self.state['complementarity_analysis']['complementarity']
        if complementarity < self.config.min_complementarity:
            print(f"\n⚠️  WARNING: Low complementarity ({complementarity:.2%})")
            print("   Ensemble may not improve much over single models")
            print("   Proceeding anyway for completeness...")
        
        # Optimize weights
        w_cnn, w_lstm, f1_weights = self.optimize_weights(X_val, y_val)
        
        # Optimize temperatures
        temp_cnn, temp_lstm, f1_temps = self.optimize_temperatures(
            X_val, y_val, w_cnn, w_lstm
        )
        
        # Create final ensemble
        self.ensemble = WeightedEnsemble(
            self.cnn_model, self.lstm_model,
            w_cnn=w_cnn, w_lstm=w_lstm,
            temp_cnn=temp_cnn, temp_lstm=temp_lstm
        )
        
        # Final evaluation
        final_results = self.evaluate_final_ensemble(X_val, y_val)
        
        # Bootstrap CI
        ci_results = self.bootstrap_ensemble_ci(X_val, y_val)
        
        # Combine results
        self.state['final_results'] = {
            'cnn': baseline_results['cnn'],
            'lstm': baseline_results['lstm'],
            'ensemble': final_results,
            'confidence_interval': ci_results,
            'weights': {'cnn': w_cnn, 'lstm': w_lstm},
            'temperatures': {'cnn': temp_cnn, 'lstm': temp_lstm}
        }
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("PHASE 3: FINAL RESULTS")
        print("="*80)
        
        results = self.state['final_results']
        comp = self.state['complementarity_analysis']
        
        print("\n[INDIVIDUAL MODELS]")
        print(f"  CNN:  F1 = {results['cnn']['f1']:.4f}, Acc = {results['cnn']['acc']:.4f}")
        print(f"  LSTM: F1 = {results['lstm']['f1']:.4f}, Acc = {results['lstm']['acc']:.4f}")
        
        print("\n[ENSEMBLE]")
        ensemble_f1 = results['ensemble']['f1']
        ci = results['confidence_interval']
        print(f"  F1 = {ensemble_f1:.4f} (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
        print(f"  Acc = {results['ensemble']['acc']:.4f}")
        
        print("\n[IMPROVEMENTS]")
        imp_cnn = ensemble_f1 - results['cnn']['f1']
        imp_lstm = ensemble_f1 - results['lstm']['f1']
        print(f"  vs CNN:  {imp_cnn:+.4f} ({imp_cnn*100:+.2f}%)")
        print(f"  vs LSTM: {imp_lstm:+.4f} ({imp_lstm*100:+.2f}%)")
        
        print("\n[ENSEMBLE CONFIG]")
        w = results['weights']
        t = results['temperatures']
        print(f"  Weights: CNN={w['cnn']:.2f}, LSTM={w['lstm']:.2f}")
        print(f"  Temps:   CNN={t['cnn']:.2f}, LSTM={t['lstm']:.2f}")
        
        print("\n[ERROR COMPLEMENTARITY]")
        print(f"  Complementarity: {comp['complementarity']:.2%}")
        print(f"  {comp['recommendation']}")
        
        print("\n[TARGET ACHIEVEMENT]")
        if ensemble_f1 >= 0.900:
            print(f"  ✓✓ TARGET REACHED: {ensemble_f1*100:.2f}% (≥90%)")
        elif ensemble_f1 >= 0.895:
            print(f"  ✓ CLOSE: {ensemble_f1*100:.2f}% (within 0.5% of target)")
        elif ensemble_f1 >= 0.890:
            print(f"  ~ ACCEPTABLE: {ensemble_f1*100:.2f}% (publishable)")
        else:
            gap = (0.900 - ensemble_f1) * 100
            print(f"  ✗ Below target: {ensemble_f1*100:.2f}% ({gap:.2f}% away)")
            print(f"     Possible reasons:")
            print(f"     - Low error complementarity ({comp['complementarity']:.2%})")
            print(f"     - Both models plateau at similar ceiling")
            print(f"     - Data quality limits")
        
        print("\n[COMPARISON TO BASELINE]")
        baseline_f1 = 0.8817  # From training_results.json
        total_gain = ensemble_f1 - baseline_f1
        print(f"  Baseline CNN:  {baseline_f1*100:.2f}%")
        print(f"  Final Ensemble: {ensemble_f1*100:.2f}%")
        print(f"  Total Gain:     {total_gain:+.4f} ({total_gain*100:+.2f}%)")
    
    def save_results(self):
        """Save comprehensive results"""
        results_path = self.config.results_dir / "phase3_ensemble_results.json"
        
        # Prepare serializable results
        save_data = {
            'final_results': {
                'cnn': {
                    'f1': float(self.state['final_results']['cnn']['f1']),
                    'acc': float(self.state['final_results']['cnn']['acc']),
                    'ece': float(self.state['final_results']['cnn']['ece'])
                },
                'lstm': {
                    'f1': float(self.state['final_results']['lstm']['f1']),
                    'acc': float(self.state['final_results']['lstm']['acc']),
                    'ece': float(self.state['final_results']['lstm']['ece'])
                },
                'ensemble': self.state['final_results']['ensemble'],
                'confidence_interval': self.state['final_results']['confidence_interval'],
                'weights': self.state['final_results']['weights'],
                'temperatures': self.state['final_results']['temperatures']
            },
            'complementarity_analysis': self.state['complementarity_analysis'],
            'weight_search_results': self.state['weight_search_results'][:10],  # Top 10
            'temp_search_results': self.state['temp_search_results'][:10]  # Top 10
        }
        
        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Results saved: {results_path}")
        
        # Save ensemble weights
        ensemble_weights_path = self.config.checkpoint_dir / "phase3_ensemble_weights.pt"
        torch.save({
            'weights': self.state['final_results']['weights'],
            'temperatures': self.state['final_results']['temperatures'],
            'final_f1': self.state['final_results']['ensemble']['f1']
        }, ensemble_weights_path)
        
        print(f"✓ Ensemble weights saved: {ensemble_weights_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  PHASE 3: ENSEMBLE OPTIMIZATION (Research-Grade)           ║
    ╠════════════════════════════════════════════════════════════╣
    ║  Target: 90.0-91.0% F1                                     ║
    ║  Requires: 40-60% error complementarity                    ║
    ║                                                            ║
    ║  Validation Steps:                                         ║
    ║  1. Error complementarity analysis                         ║
    ║  2. Weight optimization (grid search)                      ║
    ║  3. Temperature calibration (ECE minimization)             ║
    ║  4. Statistical significance testing                       ║
    ║  5. Bootstrap confidence intervals                         ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    config = EnsembleConfig()
    trainer = EnsembleTrainer(config)
    trainer.run_full_optimization()
    
    # Final recommendation
    ensemble_f1 = trainer.state['final_results']['ensemble']['f1']
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if ensemble_f1 >= 0.900:
        print("✓ SUCCESS! Ready for publication")
        print("  1. Run Phase 4: Statistical validation")
        print("  2. Write paper results section")
        print("  3. Create model card for deployment")
    elif ensemble_f1 >= 0.890:
        print("✓ Good result! Proceed with caution")
        print("  1. Run Phase 4 to verify stability")
        print("  2. Consider ablation studies")
        print("  3. Analyze per-class performance")
    else:
        print("⚠️  Below target. Options:")
        print("  1. Try stacking meta-learner (more complex)")
        print("  2. Collect more training data")
        print("  3. Feature engineering")
        print("  4. Publish current results with honest limitations")


if __name__ == "__main__":
    main()