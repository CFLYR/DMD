"""
Configuration Generator for Table 4 style single-modality FD ablation
Target setting: MOSI + GloVe + seed 1111
Experiments: (L, V, A) x (w/o FD, w/ FD)
"""
import copy
import json
from pathlib import Path

BASE_CONFIG = {
    "datasetCommonParams": {
        "dataset_root_dir": "./dataset/GloVe",
        "mosi": {
            "aligned": {
                "featurePath": "mosi_data_noalign.pkl",
                "feature_dims": [300, 5, 20],
                "train_samples": 1284,
                "num_classes": 3,
                "language": "en",
                "KeyEval": "Loss"
            },
            "unaligned": {
                "featurePath": "mosi_data_noalign.pkl",
                "feature_dims": [300, 5, 20],
                "train_samples": 1284,
                "num_classes": 3,
                "language": "en",
                "KeyEval": "Loss"
            }
        }
    },
    "dmd": {
        "commonParams": {
            "need_data_aligned": False,
            "need_model_aligned": True,
            "early_stop": 10,
            "use_bert": False,
            "use_finetune": True,
            "attn_mask": True,
            "update_epochs": 10,
            "lambda_1": 0.1,
            "lambda_2": 0.05,
            "gamma": 0.1
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
                "pretrained": "/home/.venv/lib64/python3.12/site-packages/pip/_vendor/certifi/utils/DMD/hugface",
                "epochs": 30
            }
        }
    }
}

EXPERIMENTS = [
    {"name": "table4_l_wo_fd_mosi", "single_modal": "L", "use_FD": False},
    {"name": "table4_l_w_fd_mosi", "single_modal": "L", "use_FD": True},
    {"name": "table4_v_wo_fd_mosi", "single_modal": "V", "use_FD": False},
    {"name": "table4_v_w_fd_mosi", "single_modal": "V", "use_FD": True},
    {"name": "table4_a_wo_fd_mosi", "single_modal": "A", "use_FD": False},
    {"name": "table4_a_w_fd_mosi", "single_modal": "A", "use_FD": True},
]


def build_config(exp):
    cfg = copy.deepcopy(BASE_CONFIG)
    common = cfg["dmd"]["commonParams"]
    common["single_modal"] = exp["single_modal"]

    if exp["use_FD"]:
        common["use_FD"] = True
        common["use_HomoGD"] = False
        common["use_CA"] = False
        common["use_HeteroGD"] = False
    else:
        common["use_FD"] = False
        common["use_HomoGD"] = False
        common["use_CA"] = False
        common["use_HeteroGD"] = False

    return cfg


def generate_all_configs(output_dir="experiments/ablation_study_table4/configs"):
    script_dir = Path(__file__).parent
    dmd_root = script_dir.parent
    output_path = dmd_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generate Table4 configs (MOSI + GloVe)")
    print("=" * 80)

    for exp in EXPERIMENTS:
        config = build_config(exp)
        file_path = output_path / f"{exp['name']}.json"
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ {file_path}")

    print("=" * 80)
    print(f"Generated {len(EXPERIMENTS)} configs in {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_configs()
