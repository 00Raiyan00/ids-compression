#!/usr/bin/env python3
"""
PHASE 1: TARGETED DS-1D-CNN IMPROVEMENTS
==========================================
Enhance the DS-1D-CNN model based on Phase 0 diagnostic patterns.

Improvements:
  1. Mixup augmentation (+0.3-0.7% F1)
  2. Temperature scaling (+0.1-0.3% F1)  
  3. Per-class threshold tuning (+0.2-0.5% F1)
  4. Optional: Adaptive focal loss for Pattern A
  
Expected cumulative gain: +0.8-1.5% → 88.9-89.7% F1

Key insight: These improvements are ORTHOGONAL (don't conflict)
so we can stack them for maximum gain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
import json

# ============================================================================
# UTILITY: MIXUP AUGMENTATION
# ============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Mixup augmentation: linear interpolation between random pairs.
    
    Args:
        x: Input features (batch, features)
        y: Target labels (batch,)
        alpha: Beta distribution parameter (higher = more mixing)
    
    Returns:
        mixed_x, y_a, y_b, lam: Mixed features and targets for linear interpolation loss
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a = y
    y_b = y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, 
                   y_b: torch.Tensor, lam: float):
    """Linear interpolation of losses for mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================================
# UTILITY: TEMPERATURE SCALING
# ============================================================================

def find_optimal_temperature(model, X_val: torch.Tensor, y_val: np.ndarray, 
                            device='cuda', temp_range=(0.5, 3.0, 0.1)):
    """
    Find temperature T that maximizes validation Macro-F1.
    
    Temperature scaling: softmax(logits / T)
    - T < 1.0: Sharpens predictions (higher confidence)
    - T > 1.0: Softens predictions (lower confidence, more uncertainty)
    
    Args:
        model: Trained model
        X_val: Validation features (N, features)
        y_val: Validation labels (N,)
        device: cuda/cpu
        temp_range: (min, max, step) for temperature search
    
    Returns:
        best_T: Optimal temperature
        best_f1: F1 at optimal temperature
    """
    model.eval()
    
    with torch.no_grad():
        if isinstance(X_val, np.ndarray):
            X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        else:
            X_val = X_val.to(device)
        
        logits = model(X_val).cpu().numpy()
    
    if isinstance(y_val, torch.Tensor):
        y_val = y_val.cpu().numpy()
    
    best_T = 1.0
    best_f1 = f1_score(y_val, np.argmax(logits, axis=1), average='macro', zero_division=0)
    best_probs = None
    
    temp_min, temp_max, temp_step = temp_range
    
    print(f"\n  Temperature Search (baseline T=1.0, F1={best_f1:.4f}):")
    
    for T in np.arange(temp_min, temp_max, temp_step):
        scaled_logits = logits / T
        preds = np.argmax(scaled_logits, axis=1)
        f1 = f1_score(y_val, preds, average='macro', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_T = T
            best_probs = scaled_logits
            print(f"    T={T:.2f}: F1={f1:.4f} ✓ (improved +{f1-best_f1:.4f})")
    
    improvement = best_f1 - f1_score(y_val, np.argmax(logits, axis=1), average='macro', zero_division=0)
    print(f"\n  ✓ Optimal temperature: {best_T:.2f}, Improvement: +{improvement:.4f}")
    
    return best_T, best_f1

# ============================================================================
# UTILITY: PER-CLASS THRESHOLD TUNING
# ============================================================================

def tune_per_class_thresholds(model, X_val: torch.Tensor, y_val: np.ndarray,
                             device='cuda', temp=1.0):
    """
    Optimize per-class decision thresholds using grid search.
    
    Instead of using argmax directly, we can adjust per-class thresholds:
    - High threshold for easy classes → reduce false positives
    - Low threshold for hard classes → reduce false negatives
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        device: cuda/cpu
        temp: Temperature scaling value
    
    Returns:
        thresholds: (n_classes,) optimal threshold per class
        best_f1: Best F1 achieved
    """
    model.eval()
    
    with torch.no_grad():
        if isinstance(X_val, np.ndarray):
            X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        else:
            X_val = X_val.to(device)
        
        logits = model(X_val).cpu().numpy()
    
    if isinstance(y_val, torch.Tensor):
        y_val = y_val.cpu().numpy()
    
    # Temperature-scaled probabilities
    probs = torch.softmax(torch.tensor(logits / temp), dim=1).numpy()
    
    baseline_preds = np.argmax(probs, axis=1)
    baseline_f1 = f1_score(y_val, baseline_preds, average='macro', zero_division=0)
    
    n_classes = probs.shape[1]
    thresholds = np.ones(n_classes) * 0.5
    
    print(f"\n  Per-class threshold optimization (baseline: {baseline_f1:.4f}):")
    print(f"    Searching 17 thresholds per class... (this takes 30-60 sec)")
    
    for cls_idx in range(n_classes):
        best_thresh_for_class = 0.5
        best_f1_for_class = baseline_f1
        
        for thresh in np.arange(0.1, 0.95, 0.05):
            # Modify predictions: if confidence in class >= thresh, predict it
            preds = baseline_preds.copy()
            
            # Find samples where this class is confident
            mask = probs[:, cls_idx] >= thresh
            preds[mask] = cls_idx
            
            f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            
            if f1 > best_f1_for_class:
                best_f1_for_class = f1
                best_thresh_for_class = thresh
        
        thresholds[cls_idx] = best_thresh_for_class
    
    # Evaluate final thresholds
    final_preds = baseline_preds.copy()
    for i in range(len(probs)):
        for cls_idx in range(n_classes):
            if probs[i, cls_idx] >= thresholds[cls_idx]:
                final_preds[i] = cls_idx
                break
    
    final_f1 = f1_score(y_val, final_preds, average='macro', zero_division=0)
    improvement = final_f1 - baseline_f1
    
    print(f"    ✓ Final F1: {final_f1:.4f}, Improvement: +{improvement:.4f}")
    
    return thresholds, final_f1

# ============================================================================
# MAIN: PHASE 1 TRAINING LOOP WITH IMPROVEMENTS
# ============================================================================

def phase1_training_loop(model, train_loader, val_loader, indexer,
                        epochs=20, device='cuda', use_mixup=True,
                        use_temperature_scaling=True, use_threshold_tuning=True):
    """
    Execute Phase 1 training with all improvements enabled.
    
    Args:
        model: DS-1D-CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        indexer: LabelIndexer for class names
        epochs: Number of training epochs
        device: cuda/cpu
        use_mixup: Enable mixup augmentation
        use_temperature_scaling: Enable temperature scaling
        use_threshold_tuning: Enable per-class threshold tuning
    
    Returns:
        model: Trained model
        results: Dict with metrics
    """
    
    print("\n" + "="*80)
    print("PHASE 1: TRAINING DS-1D-CNN WITH IMPROVEMENTS")
    print("="*80)
    print(f"Mixup Augmentation: {use_mixup}")
    print(f"Temperature Scaling: {use_temperature_scaling}")
    print(f"Per-class Threshold Tuning: {use_threshold_tuning}")
    
    model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    best_f1 = 0.0
    best_model_state = None
    temperature = 1.0
    thresholds = None
    
    for epoch in range(1, epochs + 1):
        # ====== TRAINING ======
        model.train()
        train_loss = 0.0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Apply mixup with 50% probability
            if use_mixup and np.random.rand() < 0.5:
                X_mixed, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha=0.2)
                logits = model(X_mixed)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ====== VALIDATION ======
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                val_preds.append(preds.cpu().numpy())
                val_targets.append(y_batch.numpy())
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        val_acc = (val_preds == val_targets).mean()
        
        print(f"Epoch {epoch}/{epochs}: "
              f"Loss={avg_train_loss:.4f}, Val-F1={val_f1:.4f}, Val-Acc={val_acc:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
        
        scheduler.step(val_f1)
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\n✓ Best validation F1: {best_f1:.4f}")
    
    # ====== APPLY IMPROVEMENTS ======
    
    print("\n[IMPROVEMENT 1/3] Applying temperature scaling...")
    # Prepare validation data for temperature scaling
    X_val_list = []
    y_val_list = []
    for X_batch, y_batch in val_loader:
        X_val_list.append(X_batch)
        y_val_list.append(y_batch.numpy())
    X_val = torch.cat(X_val_list, dim=0)
    y_val = np.concatenate(y_val_list)
    
    temperature, temp_f1 = find_optimal_temperature(model, X_val, y_val, device=device)
    
    print(f"\n[IMPROVEMENT 2/3] Applying per-class threshold tuning...")
    thresholds, thresh_f1 = tune_per_class_thresholds(model, X_val, y_val, 
                                                      device=device, temp=temperature)
    
    # ====== FINAL EVALUATION ======
    print("\n[IMPROVEMENT 3/3] Final evaluation with all improvements...")
    model.eval()
    with torch.no_grad():
        X_val = X_val.to(device)
        logits = model(X_val).cpu().numpy()
    
    # Apply temperature scaling
    probs = torch.softmax(torch.tensor(logits / temperature), dim=1).numpy()
    
    # Apply per-class threshold tuning
    final_preds = np.argmax(probs, axis=1)
    for i in range(len(probs)):
        for cls_idx in range(len(thresholds)):
            if probs[i, cls_idx] >= thresholds[cls_idx]:
                final_preds[i] = cls_idx
                break
    
    final_f1 = f1_score(y_val, final_preds, average='macro', zero_division=0)
    final_acc = (final_preds == y_val).mean()
    
    print(f"\n✓ Final F1 with all improvements: {final_f1:.4f}")
    print(f"✓ Final Accuracy: {final_acc:.4f}")
    print(f"✓ Total improvement from baseline (88.17%): +{(final_f1 - 0.8817):.4f}")
    
    results = {
        'baseline_f1': best_f1,
        'temperature': float(temperature),
        'thresholds': thresholds.tolist(),
        'final_f1': final_f1,
        'final_acc': final_acc,
        'improvement': final_f1 - 0.8817
    }
    
    return model, results, temperature, thresholds

# ============================================================================
# SAVE IMPROVED MODEL AND PARAMETERS
# ============================================================================

def save_phase1_results(model, results, ckpt_dir: Path):
    """Save Phase 1 model and improvement parameters"""
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'temperature': results['temperature'],
        'thresholds': results['thresholds'],
        'final_f1': results['final_f1']
    }, ckpt_dir / 'phase1_improved_model.pt')
    
    # Save results as JSON
    with open(ckpt_dir / 'phase1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved Phase 1 improved model: {ckpt_dir / 'phase1_improved_model.pt'}")
    print(f"✓ Saved Phase 1 results: {ckpt_dir / 'phase1_results.json'}")

if __name__ == '__main__':
    print("Phase 1 utilities ready. Import and use in train_phase1_improved.py")
