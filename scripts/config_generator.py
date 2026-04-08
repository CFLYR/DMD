"""
Configuration Generator for DMD Ablation Study
Generates 12 configurations (6 variants × 2 datasets) for Table 3 reproduction

NOTE: Using BERT (768-dim) features on UNALIGNED data only.
Paper Table 3 uses GloVe, but we deviate to use available BERT features.
Each variant has specific ablation flags (use_FD, use_HomoGD, use_CA, use_HeteroGD).
"""
import json
import os
from pathlib import Path

# Base configuration template
BASE_CONFIG = {
    "datasetCommonParams": {
        "dataset_root_dir": "./dataset",
        "mosi": {
            "aligned": {
                "featurePath": "MOSI/Processed/aligned_50.pkl",
                "feature_dims": [300, 5, 20],  # Will be updated based on use_bert
                "train_samples": 1284,
                "num_classes": 3,
                "language": "en",
                "KeyEval": "Loss"
            },
            "unaligned": {
                "featurePath": "MOSI/Processed/unaligned_50.pkl",
                "feature_dims": [300, 5, 20],
                "train_samples": 1284,
                "num_classes": 3,
                "language": "en",
                "KeyEval": "Loss"
            }
        },
        "mosei": {
            "aligned": {
                "featurePath": "MOSEI/Processed/aligned_50.pkl",
                "feature_dims": [300, 74, 35],
                "train_samples": 16326,
                "num_classes": 3,
                "language": "en",
                "KeyEval": "Loss"
            },
            "unaligned": {
                "featurePath": "MOSEI/Processed/unaligned_50.pkl",
                "feature_dims": [300, 74, 35],
                "train_samples": 16326,
                "num_classes": 3,
                "language": "en",
                "KeyEval": "Loss"
            }
        }
    },
    "dmd": {
        "commonParams": {
            "need_data_aligned": True,
            "need_model_aligned": True,
            "early_stop": 10,
            "use_bert": False,
            "use_finetune": True,
            "attn_mask": True,
            "update_epochs": 10
        },
        "datasetParams": {
            "mosi": {
                "attn_dropout_a": 0.2,
                "attn_dropout_v": 0.0,
                "relu_dropout": 0.0,
                "embed_dropout": 0.2,
                "res_dropout": 0.0,
                "dst_feature_dim_nheads": [50, 10],
                "batch_size": 16,
                "learning_rate": 0.0001,
                "nlevels": 4,
                "conv1d_kernel_size_l": 5,
                "conv1d_kernel_size_a": 5,
                "conv1d_kernel_size_v": 5,
                "text_dropout": 0.5,
                "attn_dropout": 0.3,
                "output_dropout": 0.5,
                "grad_clip": 0.6,
                "patience": 5,
                "weight_decay": 0.005,
                "transformers": "bert",
                "pretrained": "/home/.venv/lib64/python3.12/site-packages/pip/_vendor/certifi/utils/DMD/hugface"
            },
            "mosei": {
                "attn_dropout_a": 0.0,
                "attn_dropout_v": 0.0,
                "relu_dropout": 0.0,
                "embed_dropout": 0.0,
                "res_dropout": 0.0,
                "dst_feature_dim_nheads": [30, 6],
                "batch_size": 16,
                "learning_rate": 0.0001,
                "nlevels": 4,
                "conv1d_kernel_size_l": 5,
                "conv1d_kernel_size_a": 1,
                "conv1d_kernel_size_v": 3,
                "text_dropout": 0.3,
                "attn_dropout": 0.4,
                "output_dropout": 0.5,
                "grad_clip": 0.6,
                "patience": 5,
                "weight_decay": 0.001,
                "transformers": "bert",
                "pretrained": "/home/.venv/lib64/python3.12/site-packages/pip/_vendor/certifi/utils/DMD/hugface"
            }
        }
    }
}

# 6 Ablation Variants × 2 Datasets = 12 Configurations
# All use UNALIGNED data with BERT (768-dim) features
# Variants progressively remove components to measure their contribution
ABLATION_VARIANTS = {
    'variant1_full': {
        'use_FD': True,
        'use_HomoGD': True,
        'use_CA': True,
        'use_HeteroGD': True,
        'description': 'Full Model - All components active'
    },
    'variant2_no_hetero': {
        'use_FD': True,
        'use_HomoGD': True,
        'use_CA': True,
        'use_HeteroGD': False,
        'description': 'Without HeteroGD - Remove heterogeneous graph distillation'
    },
    'variant3_no_ca': {
        'use_FD': True,
        'use_HomoGD': True,
        'use_CA': False,
        'use_HeteroGD': True,
        'description': 'Without CA - Keep distillation but remove cross-modal attention'
    },
    'variant4_only_homo': {
        'use_FD': True,
        'use_HomoGD': True,
        'use_CA': False,
        'use_HeteroGD': False,
        'description': 'Only HomoGD - Feature decoupling + homogeneous GD only'
    },
    'variant5_only_fd': {
        'use_FD': True,
        'use_HomoGD': False,
        'use_CA': False,
        'use_HeteroGD': False,
        'description': 'Only FD - Feature decoupling only, no GD or CA'
    },
    'variant6_baseline': {
        'use_FD': False,
        'use_HomoGD': False,
        'use_CA': False,
        'use_HeteroGD': False,
        'description': 'Baseline - No advanced modules, direct concatenation'
    }
}

# Loss weight parameters (conditional based on active modules)
LOSS_WEIGHTS = {
    'lambda_1': 0.1,  # Decoupling loss weight (active when use_FD=True)
    'lambda_2': 0.05,  # Graph distillation loss weight (active when use_HomoGD or use_HeteroGD=True)
    'gamma': 0.1  # Orthogonality & margin weight (active when use_FD=True)
}

# Generate 12 experiments: 6 variants × 2 datasets (both unaligned)
EXPERIMENTS = []
for variant_name, variant_flags in ABLATION_VARIANTS.items():
    for dataset in ['mosi', 'mosei']:
        EXPERIMENTS.append({
            "name": f"{variant_name}_{dataset}",
            "variant": variant_name,
            "dataset": dataset,
            "aligned": False,  # All ablation experiments use UNALIGNED data
            "use_bert": True,  # All use BERT (768-dim) features
            **variant_flags
        })


def generate_config(exp_config):
    """Generate a configuration file for a specific ablation experiment"""
    import copy
    config = copy.deepcopy(BASE_CONFIG)
    
    dataset = exp_config["dataset"]
    aligned = exp_config["aligned"]
    use_bert = exp_config["use_bert"]
    
    # Update common params
    config["dmd"]["commonParams"]["need_data_aligned"] = aligned
    config["dmd"]["commonParams"]["use_bert"] = use_bert
    
    # Add ablation flags to config
    config["dmd"]["commonParams"]["use_FD"] = exp_config["use_FD"]
    config["dmd"]["commonParams"]["use_HomoGD"] = exp_config["use_HomoGD"]
    config["dmd"]["commonParams"]["use_CA"] = exp_config["use_CA"]
    config["dmd"]["commonParams"]["use_HeteroGD"] = exp_config["use_HeteroGD"]
    
    # Add loss weights
    config["dmd"]["commonParams"]["lambda_1"] = LOSS_WEIGHTS['lambda_1']
    config["dmd"]["commonParams"]["lambda_2"] = LOSS_WEIGHTS['lambda_2']
    config["dmd"]["commonParams"]["gamma"] = LOSS_WEIGHTS['gamma']
    
    # Add fixed hyperparameters for ablation study
    config["dmd"]["datasetParams"][dataset]["batch_size"] = 16
    config["dmd"]["datasetParams"][dataset]["epochs"] = 30
    config["dmd"]["datasetParams"][dataset]["learning_rate"] = 0.0001
    
    # Update feature dimensions based on use_bert
    text_dim = 768 if use_bert else 300
    alignment = "aligned" if aligned else "unaligned"
    
    config["datasetCommonParams"][dataset][alignment]["feature_dims"][0] = text_dim
    
    return config


def save_config(config, exp_name, output_dir):
    """Save configuration to JSON file"""
    output_path = Path(output_dir) / f"{exp_name}.json"
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Generated: {output_path}")
    return output_path


def generate_all_configs(output_dir="experiments/ablation_study/configs"):
    """Generate 12 ablation experiment configurations (6 variants × 2 datasets)"""
    # Get the DMD root directory
    script_dir = Path(__file__).parent
    dmd_root = script_dir.parent
    output_dir = dmd_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("DMD Ablation Study Configuration Generator")
    print("=" * 80)
    print(f"\nGenerating {len(EXPERIMENTS)} configurations (6 variants × 2 datasets)")
    print(f"Dataset: UNALIGNED MOSI & MOSEI")
    print(f"Features: BERT (768-dim)")
    print(f"Seed: 1111 (fixed)")
    print(f"Batch Size: 16")
    print(f"Epochs: 30")
    print(f"Loss Weights: λ₁={LOSS_WEIGHTS['lambda_1']}, λ₂={LOSS_WEIGHTS['lambda_2']}, γ={LOSS_WEIGHTS['gamma']}")
    print(f"\nOutput directory: {output_dir}\n")
    
    generated_files = []
    for exp in EXPERIMENTS:
        config = generate_config(exp)
        config_path = save_config(config, exp["name"], output_dir)
        generated_files.append({
            "name": exp["name"],
            "variant": exp["variant"],
            "path": config_path,
            "dataset": exp["dataset"].upper(),
            "use_FD": "✓" if exp["use_FD"] else "✗",
            "use_HomoGD": "✓" if exp["use_HomoGD"] else "✗",
            "use_CA": "✓" if exp["use_CA"] else "✗",
            "use_HeteroGD": "✓" if exp["use_HeteroGD"] else "✗",
            "description": exp["description"]
        })
    
    print("\n" + "=" * 80)
    print("Configuration Summary - Ablation Variants")
    print("=" * 80)
    print(f"{'Variant':<20} {'Dataset':<7} {'FD':<4} {'HomoGD':<8} {'CA':<4} {'HeteroGD':<10} {'Description'}")
    print("-" * 80)
    for item in generated_files:
        print(f"{item['variant']:<20} {item['dataset']:<7} {item['use_FD']:<4} {item['use_HomoGD']:<8} {item['use_CA']:<4} {item['use_HeteroGD']:<10} {item['description'][:35]}")
    
    print("\n" + "=" * 80)
    print(f"✓ Successfully generated {len(generated_files)} configuration files")
    print(f"✓ Saved to: {output_dir}")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review generated configs in", output_dir)
    print("  2. Run training: python 'run will be ablation.py' --mode train --variant <name> --dataset <mosi|mosei>")
    print("  3. Run testing: python 'run will be ablation.py' --mode test")
    print("=" * 80)
    
    return generated_files


if __name__ == "__main__":
    generate_all_configs()
