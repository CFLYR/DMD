#!/bin/bash
# Quick Start Example - DMD Experiments
# This script demonstrates typical usage patterns

echo "DMD Experiments - Quick Start Examples"
echo "========================================"
echo ""

# Example 1: Check environment
echo "Example 1: Check environment"
echo "----------------------------"
echo "Command: ./run_all.sh --check"
echo ""

# Example 2: Run smoke test
echo "Example 2: Run smoke test (2 epochs)"
echo "-------------------------------------"
echo "Command: ./run_all.sh --smoke"
echo "or"
echo "Command: python scripts/smoke_test.py --epochs 2"
echo ""

# Example 3: Train single experiment
echo "Example 3: Train single experiment"
echo "-----------------------------------"
echo "Command: python scripts/batch_train.py --experiment mosi_aligned_glove"
echo ""

# Example 4: Train all experiments
echo "Example 4: Train all experiments"
echo "---------------------------------"
echo "Command: ./run_all.sh --train"
echo "or"
echo "Command: python scripts/batch_train.py"
echo ""

# Example 5: Test all models
echo "Example 5: Test all trained models"
echo "-----------------------------------"
echo "Command: ./run_all.sh --test"
echo "or"
echo "Command: python scripts/batch_test.py"
echo ""

# Example 6: Complete pipeline
echo "Example 6: Complete pipeline (smoke + train + test)"
echo "---------------------------------------------------"
echo "Command: ./run_all.sh --all"
echo ""

# Example 7: Continue from interrupted training
echo "Example 7: Continue from interrupted training"
echo "----------------------------------------------"
echo "If training stopped at experiment 3 (0-based index 2):"
echo "Command: python scripts/batch_train.py --continue-from 2"
echo ""

# Example 8: Check results
echo "Example 8: Check results"
echo "------------------------"
echo "View training results:"
echo "  cat experiments/results/training_results.json"
echo ""
echo "View comparison with paper:"
echo "  cat experiments/results/comparison_with_paper.csv"
echo ""
echo "View logs:"
echo "  tail -f experiments/logs/mosi_aligned_glove.log"
echo ""

echo "For full documentation, see experiments/README.md"
