#!/usr/bin/env python3
"""
PHASE 0: PER-CLASS FAILURE ANALYSIS
=====================================
Identify which classes drag down F1 from 88.17% → 90%.

Outputs:
  - Per-class metrics (F1, Precision, Recall, Support)
  - Confusion matrix heatmap
  - Misclassification patterns (top 15)
  - Class difficulty ranking
  - Targeted recommendations

Time: ~4 hours execution + analysis
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, precision_recall_fscore_support
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD CHECKPOINT AND DATA
# ============================================================================

def load_checkpoint(ckpt_dir: Path, device='cuda'):
    """Load best model checkpoint"""
    ckpt = torch.load(ckpt_dir / 'best_model.pt', map_location=device)
    
    # Assuming EdgeIDSNetV2 architecture
    from train_76m_streaming_fixed import EdgeIDSNetV2, LabelIndexer
    
    # Load indexer
    indexer = LabelIndexer.load(ckpt_dir.parent / 'label_indexer.pkl')
    
    # Create model
    n_features = 41
    n_classes = indexer.n_classes()
    model = EdgeIDSNetV2(n_features, n_classes, base_filters=256, dropout=0.4)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {ckpt['epoch']}")
    print(f"  Best Val F1: {ckpt['training_state']['best_val_f1']:.4f}")
    print(f"  Classes: {n_classes}")
    
    return model, indexer, ckpt

def load_validation_data(ckpt_dir: Path):
    """Load validation data"""
    val_data = torch.load(ckpt_dir / 'validation_data.pt')
    X_val = val_data['X_val']  # torch.Tensor
    y_val = val_data['y_val']  # torch.Tensor
    
    print(f"✓ Loaded validation data: {X_val.shape}")
    return X_val, y_val

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

def predict_all(model, X_val, device='cuda', batch_size=512):
    """Generate predictions on full validation set"""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_logits = []
    
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            Xv = X_val[i:i+batch_size].to(device) if isinstance(X_val, torch.Tensor) else \
                 torch.tensor(X_val[i:i+batch_size], dtype=torch.float32, device=device)
            
            logits = model(Xv)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_logits = np.concatenate(all_logits)
    
    return all_preds, all_probs, all_logits

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_per_class_metrics(y_true, y_pred, y_probs, indexer):
    """
    Compute per-class metrics and identify problem classes.
    
    Returns:
      - DataFrame with per-class metrics
      - Lists of classes by difficulty pattern
    """
    
    # Get classification report
    report = classification_report(
        y_true, y_pred,
        target_names=list(indexer.idx_to_class.values()),
        output_dict=True,
        zero_division=0
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(report).T
    df = df[~df.index.str.contains('accuracy|macro|weighted', case=False)]
    df = df.drop(['support'], axis=1) if 'support' in df.columns else df
    df = df.astype(float)
    
    # Sort by F1
    df = df.sort_values('f1-score', ascending=True)
    
    # Identify patterns
    very_poor = df[df['f1-score'] < 0.80].index.tolist()     # F1 < 80%
    poor = df[(df['f1-score'] >= 0.80) & (df['f1-score'] < 0.85)].index.tolist()  # 80-85%
    mediocre = df[(df['f1-score'] >= 0.85) & (df['f1-score'] < 0.90)].index.tolist()  # 85-90%
    good = df[df['f1-score'] >= 0.90].index.tolist()  # > 90%
    
    return df, {'very_poor': very_poor, 'poor': poor, 'mediocre': mediocre, 'good': good}

def analyze_confusion_matrix(y_true, y_pred, indexer):
    """
    Analyze confusion matrix to find misclassification patterns.
    
    Returns:
      - Confusion matrix
      - Top 15 misclassifications
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract misclassifications
    misclass_list = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                true_class = indexer.idx_to_class[i]
                pred_class = indexer.idx_to_class[j]
                count = cm[i, j]
                pct_of_true = (count / cm[i].sum() * 100) if cm[i].sum() > 0 else 0
                
                misclass_list.append({
                    'True Class': true_class,
                    'Predicted Class': pred_class,
                    'Count': count,
                    '% of True': pct_of_true
                })
    
    # Sort by count
    misclass_df = pd.DataFrame(misclass_list).sort_values('Count', ascending=False).head(15)
    
    return cm, misclass_df

def analyze_probability_calibration(y_true, y_probs, indexer):
    """
    Analyze if model confidence aligns with correctness.
    For correct predictions: max probability should be high.
    For incorrect predictions: max probability should be low (ideally).
    """
    
    preds = np.argmax(y_probs, axis=1)
    max_probs = np.max(y_probs, axis=1)
    
    correct = (preds == y_true)
    
    correct_conf = max_probs[correct]
    incorrect_conf = max_probs[~correct]
    
    stats = {
        'Correct Predictions - Mean Confidence': correct_conf.mean(),
        'Correct Predictions - Median Confidence': np.median(correct_conf),
        'Incorrect Predictions - Mean Confidence': incorrect_conf.mean(),
        'Incorrect Predictions - Median Confidence': np.median(incorrect_conf),
        'Confidence Gap (Correct - Incorrect)': correct_conf.mean() - incorrect_conf.mean(),
    }
    
    return stats

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("\n" + "="*80)
    print("PHASE 0: PER-CLASS FAILURE ANALYSIS")
    print("="*80)
    
    # Paths
    ckpt_dir = Path('checkpoints/pytorch_fixed_full')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load checkpoint
    print("\n[1/5] Loading checkpoint...")
    model, indexer, ckpt = load_checkpoint(ckpt_dir, device=device)
    
    # Load validation data
    print("[2/5] Loading validation data...")
    X_val, y_val = load_validation_data(ckpt_dir)
    y_val_np = y_val.numpy()
    
    # Generate predictions
    print("[3/5] Generating predictions (this may take 2-3 min)...")
    y_pred, y_probs, y_logits = predict_all(model, X_val, device=device)
    
    # Per-class metrics
    print("[4/5] Computing per-class metrics...")
    df_metrics, patterns = analyze_per_class_metrics(y_val_np, y_pred, y_probs, indexer)
    
    # Confusion matrix
    print("[5/5] Analyzing confusion matrix...")
    cm, misclass_df = analyze_confusion_matrix(y_val_np, y_pred, indexer)
    
    # Probability calibration
    calib_stats = analyze_probability_calibration(y_val_np, y_probs, indexer)
    
    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("RESULTS: PER-CLASS PERFORMANCE (Sorted by F1, Worst First)")
    print("="*80)
    print(df_metrics[['precision', 'recall', 'f1-score']].to_string())
    
    # Overall metrics
    overall_f1 = f1_score(y_val_np, y_pred, average='macro', zero_division=0)
    overall_acc = (y_pred == y_val_np).mean()
    print(f"\nOverall Macro-F1: {overall_f1:.4f}")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    
    # Class patterns
    print("\n" + "="*80)
    print("CLASS DIFFICULTY PATTERNS")
    print("="*80)
    print(f"Very Poor (F1 < 80%): {len(patterns['very_poor'])} classes")
    if patterns['very_poor']:
        for cls in patterns['very_poor']:
            f1 = df_metrics.loc[cls, 'f1-score']
            print(f"  • {cls:25s}: F1={f1:.4f} (CRITICAL - needs adaptive focal)")
    
    print(f"\nPoor (80% ≤ F1 < 85%): {len(patterns['poor'])} classes")
    if patterns['poor']:
        for cls in patterns['poor']:
            f1 = df_metrics.loc[cls, 'f1-score']
            print(f"  • {cls:25s}: F1={f1:.4f} (HIGH PRIORITY)")
    
    print(f"\nMediaocre (85% ≤ F1 < 90%): {len(patterns['mediocre'])} classes")
    if patterns['mediocre']:
        for cls in patterns['mediocre']:
            f1 = df_metrics.loc[cls, 'f1-score']
            print(f"  • {cls:25s}: F1={f1:.4f} (medium priority)")
    
    print(f"\nGood (F1 ≥ 90%): {len(patterns['good'])} classes")
    
    # Misclassifications
    print("\n" + "="*80)
    print("TOP 15 MISCLASSIFICATION PATTERNS (What's confusing the model?)")
    print("="*80)
    print(misclass_df.to_string(index=False))
    
    # Confidence analysis
    print("\n" + "="*80)
    print("CONFIDENCE CALIBRATION (Is model confident when correct?)")
    print("="*80)
    for key, val in calib_stats.items():
        print(f"  {key:45s}: {val:.4f}")
    
    # ========================================================================
    # DIAGNOSTIC VERDICT
    # ========================================================================
    
    print("\n" + "="*80)
    print("DIAGNOSTIC VERDICT: Which Phase 1 Strategy to Use?")
    print("="*80)
    
    n_very_poor = len(patterns['very_poor'])
    n_poor = len(patterns['poor'])
    n_mediocre = len(patterns['mediocre'])
    
    gap = 0.90 - overall_f1
    
    if n_very_poor >= 2:
        print(f"\n✓ PATTERN A DETECTED: {n_very_poor} classes with F1 < 80%")
        print("  Strategy: Adaptive focal loss + class-specific gamma")
        print(f"  Expected Gain: +0.5-1.0% → {overall_f1 + 0.75:.4f} F1")
        print("  Action: Implement AdaptiveCBFocalLoss in Phase 1")
        
    elif n_poor >= 3:
        print(f"\n✓ PATTERN B DETECTED: {n_poor} classes with 80-85% F1")
        print("  Strategy: Mixup augmentation + focal loss + threshold tuning")
        print(f"  Expected Gain: +0.5-1.0% → {overall_f1 + 0.75:.4f} F1")
        print("  Action: Implement mixup in Phase 1")
        
    else:
        print(f"\n✓ PATTERN C DETECTED: All classes > 80%, F1 variance issue")
        print("  Strategy: Temperature scaling + threshold tuning")
        print(f"  Expected Gain: +0.2-0.5% → {overall_f1 + 0.35:.4f} F1")
        print("  Action: Implement temperature scaling in Phase 1")
    
    print(f"\nGap to 90%: {gap:.4f} ({gap*100:.2f} percentage points)")
    print(f"Estimated reachable (Phase 1): {overall_f1 + 0.5:.4f} - {overall_f1 + 1.0:.4f}")
    print(f"LSTM Expected: 89.0 - 89.8%")
    print(f"Ensemble Expected: 90.0 - 91.0%")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    df_metrics.to_csv(results_dir / 'phase0_per_class_metrics.csv')
    misclass_df.to_csv(results_dir / 'phase0_top_misclassifications.csv', index=False)
    
    # Save diagnostics to text
    with open(results_dir / 'phase0_diagnostics.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 0 DIAGNOSTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Macro-F1: {overall_f1:.4f}\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n")
        f.write(f"Gap to 90%: {gap:.4f}\n\n")
        f.write("Class Patterns:\n")
        f.write(f"  Very Poor (F1<80%): {len(patterns['very_poor'])}\n")
        f.write(f"  Poor (80-85% F1): {len(patterns['poor'])}\n")
        f.write(f"  Mediocre (85-90% F1): {len(patterns['mediocre'])}\n")
        f.write(f"  Good (>90% F1): {len(patterns['good'])}\n")
        f.write("\n" + str(df_metrics[['precision', 'recall', 'f1-score']]))
    
    print(f"\n✓ Results saved to {results_dir}")
    print(f"  - phase0_per_class_metrics.csv")
    print(f"  - phase0_top_misclassifications.csv")
    print(f"  - phase0_diagnostics.txt")

if __name__ == '__main__':
    main()
