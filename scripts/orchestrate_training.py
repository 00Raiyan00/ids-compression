#!/usr/bin/env python3
"""
UNIFIED TRAINING ORCHESTRATION: Phase 0 → 1 → 2 → 3
=====================================================

This script orchestrates the complete 90%+ F1 improvement pipeline:
  
  Phase 0: Analyze which classes drag down F1 (4 hours execution)
  Phase 1: Improve DS-1D-CNN with mixup + temperature scaling (2 days)
  Phase 2: Train LSTM model (2 days)
  Phase 3: Ensemble CNN + LSTM (1 day)
  
Expected Timeline: 5-6 days total
Expected Result: 90.0-91.0% Macro-F1

Usage:
  python scripts/orchestrate_training.py --phase 0  # Run Phase 0
  python scripts/orchestrate_training.py --phase 1  # Run Phase 1 (on top of Phase 0 results)
  python scripts/orchestrate_training.py --phase 2  # Run Phase 2
  python scripts/orchestrate_training.py --phase 3  # Run Phase 3
  python scripts/orchestrate_training.py --phase all # Run all phases sequentially
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import json

# ============================================================================
# PHASE 0: ANALYSIS
# ============================================================================

def run_phase0():
    """Execute per-class analysis"""
    print("\n" + "="*80)
    print("RUNNING PHASE 0: Per-Class Failure Analysis")
    print("="*80)
    print("This analyzes which classes drag down F1 and determines improvement strategy.")
    print("\nRun manually:")
    print("  python scripts/phase0_per_class_analysis.py\n")
    
    # Try to import and run
    try:
        from scripts.phase0_per_class_analysis import main as phase0_main
        phase0_main()
        print("\n✓ Phase 0 complete. Check results/ directory for analysis.")
        return True
    except Exception as e:
        print(f"\n✗ Phase 0 failed: {e}")
        print("  Fix: Ensure checkpoints/pytorch_fixed_full/ exists with best_model.pt")
        return False

# ============================================================================
# PHASE 1: DS-1D-CNN IMPROVEMENTS
# ============================================================================

def run_phase1():
    """Execute DS-1D-CNN improvements"""
    print("\n" + "="*80)
    print("RUNNING PHASE 1: DS-1D-CNN Improvements")
    print("="*80)
    print("This applies: Mixup + Temperature Scaling + Threshold Tuning")
    print("\nRun manually:")
    print("  python scripts/train_phase1_improved.py\n")
    
    try:
        from scripts.train_phase1_improved import main as phase1_main
        phase1_main()
        print("\n✓ Phase 1 complete. Improved model saved.")
        return True
    except Exception as e:
        print(f"\n✗ Phase 1 failed: {e}")
        return False

# ============================================================================
# PHASE 2: LSTM TRAINING
# ============================================================================

def run_phase2():
    """Execute LSTM training"""
    print("\n" + "="*80)
    print("RUNNING PHASE 2: LSTM Training")
    print("="*80)
    print("This trains bidirectional LSTM with temporal modeling")
    print("\nRun manually:")
    print("  python scripts/train_phase2_lstm.py\n")
    
    try:
        from scripts.train_phase2_lstm import main as phase2_main
        phase2_main()
        print("\n✓ Phase 2 complete. LSTM model trained and saved.")
        return True
    except Exception as e:
        print(f"\n✗ Phase 2 failed: {e}")
        return False

# ============================================================================
# PHASE 3: ENSEMBLE
# ============================================================================

def run_phase3():
    """Execute ensemble optimization"""
    print("\n" + "="*80)
    print("RUNNING PHASE 3: Ensemble Optimization")
    print("="*80)
    print("This combines CNN + LSTM via weighted ensemble")
    print("\nRun manually:")
    print("  python scripts/train_phase3_ensemble.py\n")
    
    try:
        from scripts.train_phase3_ensemble import main as phase3_main
        phase3_main()
        print("\n✓ Phase 3 complete. Ensemble trained and optimized.")
        return True
    except Exception as e:
        print(f"\n✗ Phase 3 failed: {e}")
        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Orchestrate 90%+ F1 improvement pipeline")
    parser.add_argument('--phase', type=str, default='all',
                       choices=['0', '1', '2', '3', 'all'],
                       help='Which phase to run')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("90%+ MACRO-F1 IMPROVEMENT PIPELINE")
    print("="*80)
    print(f"Target: Macro-F1 ≥ 90%")
    print(f"Current: 88.17%")
    print(f"Gap: 1.83 percentage points")
    print(f"\nStrategy: CNN + LSTM Ensemble")
    print(f"Timeline: 5-6 days")
    
    phases = []
    if args.phase == 'all':
        phases = ['0', '1', '2', '3']
    else:
        phases = [args.phase]
    
    results = {}
    
    for phase in phases:
        if phase == '0':
            results['phase0'] = run_phase0()
        elif phase == '1':
            results['phase1'] = run_phase1()
        elif phase == '2':
            results['phase2'] = run_phase2()
        elif phase == '3':
            results['phase3'] = run_phase3()
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    for phase, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {phase.upper()}: {'Complete' if success else 'Failed'}")
    
    all_success = all(results.values())
    if all_success:
        print("\n✓ All phases complete!")
        print("\nExpected results:")
        print("  DS-1D-CNN improved: 88.9-89.5% F1")
        print("  LSTM model:         89.0-89.8% F1")
        print("  Ensemble:           90.0-91.0% F1")
    else:
        print("\n✗ Some phases failed. Check logs above.")
    
    return 0 if all_success else 1

if __name__ == '__main__':
    sys.exit(main())
