#!/usr/bin/env python3
"""
TRAIN LSTM IDS MODEL (Phase 2)
================================

Trains research-grade LSTM model with:
  - Bidirectional architecture
  - 2 layers for hierarchical learning
  - Proper Xavier/Orthogonal weight initialization
  - Temporal feature engineering
  - Gradient clipping
  - Learning rate scheduling
  
Expected F1: 89.0-89.8% (better than CNN on temporal patterns)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
import json
from tqdm import tqdm

def main():
    print("\n" + "="*80)
    print("PHASE 2: TRAINING LSTM IDS MODEL")
    print("="*80)
    
    # ========== CONFIGURATION ==========
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 256
    EPOCHS = 25
    SEED = 42
    CKPT_DIR = Path('checkpoints/pytorch_fixed_full')
    
    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    
    # ========== LOAD DATA ==========
    print("\n[1/5] Loading validation data...")
    try:
        val_data = torch.load(CKPT_DIR / 'validation_data.pt', map_location=DEVICE)
    except FileNotFoundError:
        print("  ✗ validation_data.pt not found in checkpoints directory")
        print("    Run: python scripts/prepare_validation_data.py or ensure checkpoints exist")
        return
    X_val = val_data['X_val']
    y_val = val_data['y_val']
    
    # Load training data
    print("[2/5] Loading training data...")
    try:
        train_data = torch.load(CKPT_DIR / 'train_data.pt', map_location=DEVICE)
    except FileNotFoundError:
        print("  ✗ train_data.pt not found in checkpoints directory")
        print("    Run: python scripts/prepare_training_data.py or ensure checkpoints exist")
        return
    X_train = train_data['X_train']
    y_train = train_data['y_train']
    
    # Ensure tensors (allow numpy arrays in checkpoints)
    if isinstance(X_train, np.ndarray):
        X_train = torch.from_numpy(X_train).float()
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train).long()
    if isinstance(X_val, np.ndarray):
        X_val = torch.from_numpy(X_val).float()
    if isinstance(y_val, np.ndarray):
        y_val = torch.from_numpy(y_val).long()

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  y_val shape: {y_val.shape}")
    
    # ========== CREATE DATA LOADERS ==========
    print("[3/5] Creating data loaders...")
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # ========== CREATE LSTM MODEL ==========
    print("[4/5] Building LSTM model...")
    from models.lstm_ids_v2 import LSTMIDSv2, train_lstm_model, save_lstm_model
    
    lstm_model = LSTMIDSv2(
        n_features=41,
        n_classes=18,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        use_attention=False
    )
    
    n_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"  LSTM parameters: {n_params:,}")
    
    # ========== TRAIN MODEL ==========
    print("[5/5] Training LSTM model (this takes 5-10 minutes)...")
    metrics = train_lstm_model(
        lstm_model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        device=DEVICE,
        use_mixup=True
    )
    
    # ========== SAVE MODEL ==========
    print("\nSaving LSTM model...")
    save_lstm_model(lstm_model, metrics, CKPT_DIR)
    
    # ========== FINAL EVALUATION ==========
    print("\n" + "="*80)
    print("PHASE 2: RESULTS")
    print("="*80)
    print(f"LSTM Macro-F1: {metrics['final_f1']:.4f}")
    print(f"LSTM Accuracy: {metrics['final_acc']:.4f}")
    print(f"Improvement over baseline CNN: +{metrics['improvement_over_cnn']:.4f}")
    
    # Compare with CNN
    print("\nComparison:")
    print(f"  DS-1D-CNN baseline:  88.17%")
    print(f"  LSTM (Phase 2):      {metrics['final_f1']*100:.2f}%")
    if metrics['final_f1'] > 0.8817:
        print(f"  LSTM is better by:   +{(metrics['final_f1']-0.8817)*100:.2f}%")
    
    print("\nNext step: Phase 3 Ensemble")
    print("Expected ensemble F1: 90.0-91.0%")

if __name__ == '__main__':
    main()
