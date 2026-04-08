"""
Batch Training Script for DMD Ablation Study
Trains a single variant with specified dataset
Usage: python "batch_train_will be ablation.py" --variant variant1_full --dataset mosi
"""
import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
dmd_root = script_dir.parent
sys.path.insert(0, str(dmd_root))

from run import DMD_run


def train_variant(variant_name, dataset_name, config_dir="experiments/ablation_study/configs",
                  model_dir="experiments/ablation_study/models", log_dir="experiments/ablation_study/logs",
                  epochs=None):
    """
    Train a single ablation variant
    
    Args:
        variant_name: Name of variant (e.g., 'variant1_full')
        dataset_name: Dataset name ('mosi' or 'mosei')
        config_dir: Directory containing config files
        model_dir: Directory to save models
        log_dir: Directory to save logs
        epochs: Override epoch count (default: 30 from config)
    """
    # Construct paths
    config_file = dmd_root / config_dir / f"{variant_name}_{dataset_name}.json"
    model_save_dir = dmd_root / model_dir / f"{variant_name}_{dataset_name}"
    log_save_dir = dmd_root / log_dir
    
    # Create directories
    model_save_dir.mkdir(parents=True, exist_ok=True)
    log_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify config exists
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    print("=" * 80)
    print(f"DMD Ablation Training - {variant_name} on {dataset_name.upper()}")
    print("=" * 80)
    print(f"Config: {config_file}")
    print(f"Model Output: {model_save_dir}")
    print(f"Log Output: {log_save_dir}")
    if epochs:
        print(f"Override Epochs: {epochs}")
    print("=" * 80)
    
    # Load config to display settings
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    ablation_flags = config['dmd']['commonParams']
    print(f"\nAblation Settings:")
    print(f"  use_FD: {ablation_flags.get('use_FD', True)}")
    print(f"  use_HomoGD: {ablation_flags.get('use_HomoGD', True)}")
    print(f"  use_CA: {ablation_flags.get('use_CA', True)}")
    print(f"  use_HeteroGD: {ablation_flags.get('use_HeteroGD', True)}")
    print(f"  lambda_1: {ablation_flags.get('lambda_1', 0.1)}")
    print(f"  lambda_2: {ablation_flags.get('lambda_2', 0.05)}")
    print("=" * 80 + "\n")
    
    # Run training
    try:
        # Build DMD_run arguments
        dmd_args = {
            'model_name': 'dmd',
            'dataset_name': dataset_name,
            'config_file': str(config_file),
            'seeds': [1111],  # Fixed seed for ablation study
            'model_save_dir': str(model_save_dir),
            'res_save_dir': str(model_save_dir / "results"),
            'log_dir': str(log_save_dir),
            'mode': 'train',
            'is_distill': True
        }
        
        # Override epochs if provided
        if epochs:
            dmd_args['epochs'] = epochs
            print(f"\n{'='*80}")
            print(f"DEBUG: Epochs Override Requested")
            print(f"  epochs value: {epochs}")
            print(f"  dmd_args['epochs']: {dmd_args.get('epochs', 'NOT SET')}")
            print(f"{'='*80}\n")
        
        DMD_run(**dmd_args)
        
        print("\n" + "=" * 80)
        print(f"✓ Training completed successfully!")
        print(f"✓ Model saved to: {model_save_dir}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Training failed with error:")
        print(f"  {str(e)}")
        print("=" * 80)
        raise


def main():
    parser = argparse.ArgumentParser(description='Train DMD ablation variant')
    parser.add_argument('--variant', type=str, required=True,
                       choices=['variant1_full', 'variant2_no_hetero', 'variant3_no_ca',
                               'variant4_only_homo', 'variant5_only_fd', 'variant6_baseline'],
                       help='Variant to train')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['mosi', 'mosei'],
                       help='Dataset to use')
    parser.add_argument('--config-dir', type=str, default='experiments/ablation_study/configs',
                       help='Config directory')
    parser.add_argument('--model-dir', type=str, default='experiments/ablation_study/models',
                       help='Model output directory')
    parser.add_argument('--log-dir', type=str, default='experiments/ablation_study/logs',
                       help='Log output directory')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epoch count (default: 30 from config). Use 1 for smoke test.')
    
    args = parser.parse_args()
    
    train_variant(
        variant_name=args.variant,
        dataset_name=args.dataset,
        config_dir=args.config_dir,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
