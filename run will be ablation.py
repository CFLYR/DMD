"""
Main entry for Table 4 single-modality FD ablation on MOSI/MOSEI + GloVe
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TRAIN_SCRIPT = ROOT / "scripts" / "batch_train_will be ablation.py"
TEST_SCRIPT = ROOT / "scripts" / "batch_test will be ablation.py"
CONFIG_SCRIPT = ROOT / "scripts" / "config_generator will be ablation.py"

VARIANTS = [
    "table4_l_wo_fd_mosi",
    "table4_l_w_fd_mosi",
    "table4_v_wo_fd_mosi",
    "table4_v_w_fd_mosi",
    "table4_a_wo_fd_mosi",
    "table4_a_w_fd_mosi",
    "table4_l_wo_fd_mosei",
    "table4_l_w_fd_mosei",
    "table4_v_wo_fd_mosei",
    "table4_v_w_fd_mosei",
    "table4_a_wo_fd_mosei",
    "table4_a_w_fd_mosei",
]


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def train_one(variant, epochs=None):
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--variant", variant]
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    return run_cmd(cmd)


def train_all(epochs=None):
    ok = True
    for v in VARIANTS:
        if not train_one(v, epochs=epochs):
            ok = False
    return ok


def test_all():
    return run_cmd([sys.executable, str(TEST_SCRIPT)])


def gen_configs():
    return run_cmd([sys.executable, str(CONFIG_SCRIPT)])


def main():
    parser = argparse.ArgumentParser(description="Table4 ablation runner")
    parser.add_argument("--mode", required=True, choices=["gen", "train", "test"])
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "gen":
        ok = gen_configs()
    elif args.mode == "train":
        if args.all:
            ok = train_all(epochs=args.epochs)
        else:
            if not args.variant:
                parser.error("--variant is required when --all is not set")
            ok = train_one(args.variant, epochs=args.epochs)
    else:
        ok = test_all()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
