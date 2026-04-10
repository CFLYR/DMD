"""
Batch Training Script for Table 4 single-modality FD ablation
"""
import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).parent
dmd_root = script_dir.parent
sys.path.insert(0, str(dmd_root))

from run import DMD_run

VARIANTS = [
    "table4_l_wo_fd_mosi",
    "table4_l_w_fd_mosi",
    "table4_v_wo_fd_mosi",
    "table4_v_w_fd_mosi",
    "table4_a_wo_fd_mosi",
    "table4_a_w_fd_mosi",
]


def train_variant(
    variant_name,
    config_dir="experiments/ablation_study_table4/configs",
    model_dir="experiments/ablation_study_table4/models",
    log_dir="experiments/ablation_study_table4/logs",
    epochs=None,
):
    config_file = dmd_root / config_dir / f"{variant_name}.json"
    model_save_dir = dmd_root / model_dir / variant_name
    log_save_dir = dmd_root / log_dir
    model_save_dir.mkdir(parents=True, exist_ok=True)
    log_save_dir.mkdir(parents=True, exist_ok=True)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        cfg = json.load(f)
    cmn = cfg["dmd"]["commonParams"]

    print("=" * 80)
    print(f"Training {variant_name}")
    print(f"Config: {config_file}")
    print(f"Model dir: {model_save_dir}")
    print(f"Log dir: {log_save_dir}")
    print(f"single_modal: {cmn.get('single_modal', 'LAV')}, use_FD: {cmn.get('use_FD', True)}")
    print("=" * 80)

    kwargs = {
        "model_name": "dmd",
        "dataset_name": "mosi",
        "config_file": str(config_file),
        "seeds": [1111],
        "model_save_dir": str(model_save_dir),
        "res_save_dir": str(model_save_dir / "results"),
        "log_dir": str(log_save_dir),
        "mode": "train",
        "is_distill": True,
    }
    if epochs is not None:
        kwargs["epochs"] = epochs

    DMD_run(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Train Table4 ablation variant")
    parser.add_argument("--variant", type=str, required=True, choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config-dir", type=str, default="experiments/ablation_study_table4/configs")
    parser.add_argument("--model-dir", type=str, default="experiments/ablation_study_table4/models")
    parser.add_argument("--log-dir", type=str, default="experiments/ablation_study_table4/logs")
    args = parser.parse_args()

    train_variant(
        variant_name=args.variant,
        config_dir=args.config_dir,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
