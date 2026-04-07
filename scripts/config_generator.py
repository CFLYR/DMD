"""
Configuration Generator for DMD Experiments
Generates 2 BERT aligned experiment configurations for reproducing Table 1 & 2 from the paper

NOTE: Data files only contain BERT (768-dim) features, not GloVe (300-dim).
The 'text' key in .pkl files stores BERT features, despite the naming.
Only the 2 BERT aligned experiments reported in the paper can be reproduced.
Unaligned experiments are not in the paper, so they are excluded.
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

# 2 BERT aligned experiment configurations - exactly matching paper Table 1 & 2 (rows with *)
# NOTE: Data files only contain BERT features (768-dim), no GloVe (300-dim)
# The 'text' key in .pkl files stores 768-dim BERT, not 300-dim GloVe
# Paper only reports Aligned BERT results, so we only reproduce those 2 experiments
EXPERIMENTS = [
    {
        "name": "mosi_aligned_bert",
        "dataset": "mosi",
        "aligned": True,
        "use_bert": True,
        "expected_acc7": 45.6,
        "table": "Table 1 - DMD (Ours)*"
    },
    {
        "name": "mosei_aligned_bert",
        "dataset": "mosei",
        "aligned": True,
        "use_bert": True,
        "expected_acc7": 54.5,
        "table": "Table 2 - DMD (Ours)*"
    }
]


def generate_config(exp_config):
    """Generate a configuration file for a specific experiment"""
    import copy
    config = copy.deepcopy(BASE_CONFIG)
    
    dataset = exp_config["dataset"]
    aligned = exp_config["aligned"]
    use_bert = exp_config["use_bert"]
    
    # Update common params
    config["dmd"]["commonParams"]["need_data_aligned"] = aligned
    config["dmd"]["commonParams"]["use_bert"] = use_bert
    
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


def generate_all_configs(output_dir="experiments/configs"):
    """Generate 2 BERT aligned experiment configurations"""
    # Get the DMD root directory
    script_dir = Path(__file__).parent
    dmd_root = script_dir.parent
    output_dir = dmd_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DMD Experiment Configuration Generator (BERT Aligned Only)")
    print("=" * 70)
    print(f"\nNOTE: Data files only contain BERT features (768-dim)")
    print(f"      Only aligned experiments from paper tables are reproduced\n")
    print(f"Generating configurations for {len(EXPERIMENTS)} experiments...")
    print(f"Output directory: {output_dir}\n")
    
    generated_files = []
    for exp in EXPERIMENTS:
        config = generate_config(exp)
        config_path = save_config(config, exp["name"], output_dir)
        generated_files.append({
            "name": exp["name"],
            "path": config_path,
            "dataset": exp["dataset"].upper(),
            "aligned": "Aligned" if exp["aligned"] else "Unaligned",
            "feature": "BERT (768d)" if exp["use_bert"] else "GloVe (300d)",
            "expected_acc7": exp["expected_acc7"],
            "table": exp["table"]
        })
    
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"{'Experiment':<25} {'Dataset':<8} {'Aligned':<10} {'Feature':<15} {'Expected ACC7'}")
    print("-" * 70)
    for item in generated_files:
        exp_acc = f"{item['expected_acc7']:.1f}%" if item['expected_acc7'] else "N/A"
        print(f"{item['name']:<25} {item['dataset']:<8} {item['aligned']:<10} {item['feature']:<15} {exp_acc}")
    
    print("\n" + "=" * 70)
    print(f"✓ Successfully generated {len(generated_files)} configuration files")
    print("=" * 70)
    
    return generated_files


if __name__ == "__main__":
    generate_all_configs()
