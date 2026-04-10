"""
Batch Testing Script for Table 4 single-modality FD ablation (MOSEI only)
"""
import csv
import sys
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent
dmd_root = script_dir.parent
sys.path.insert(0, str(dmd_root))

from run import DMD_run

VARIANTS = [
    ("L", "table4_l_wo_fd_mosei", "w/o FD"),
    ("L", "table4_l_w_fd_mosei", "w/ FD"),
    ("V", "table4_v_wo_fd_mosei", "w/o FD"),
    ("V", "table4_v_w_fd_mosei", "w/ FD"),
    ("A", "table4_a_wo_fd_mosei", "w/o FD"),
    ("A", "table4_a_w_fd_mosei", "w/ FD"),
]


def test_variant(
    variant_name,
    config_dir="experiments/ablation_study_table4/configs",
    model_dir="experiments/ablation_study_table4/models",
    results_dir="experiments/ablation_study_table4/results",
):
    config_file = dmd_root / config_dir / f"{variant_name}.json"
    model_path = dmd_root / model_dir / variant_name / "dmd-mosei.pth"
    results_save_dir = dmd_root / results_dir
    results_save_dir.mkdir(parents=True, exist_ok=True)

    if (not config_file.exists()) or (not model_path.exists()):
        return None

    return DMD_run(
        model_name="dmd",
        dataset_name="mosei",
        config_file=str(config_file),
        seeds=[1111],
        model_save_dir=str(model_path.parent),
        res_save_dir=str(results_save_dir),
        log_dir=str(results_save_dir.parent / "logs"),
        mode="test",
        is_distill=False,
    )


def generate_table4_csv(rows, output_file="experiments/ablation_study_table4/table4_results.csv"):
    output_path = dmd_root / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Modality", "FD", "Acc_2", "F1_score", "Acc_7", "MAE"],
        )
        writer.writeheader()
        writer.writerows(rows)
        
        wo_fd_metrics = {
            "Acc_2": [], "F1_score": [], "Acc_7": [], "MAE": []
        }
        w_fd_metrics = {
            "Acc_2": [], "F1_score": [], "Acc_7": [], "MAE": []
        }
        
        for row in rows:
            if row["FD"] == "w/o FD":
                for metric in wo_fd_metrics:
                    val = row.get(metric, "N/A")
                    if val != "N/A":
                        wo_fd_metrics[metric].append(float(val))
            elif row["FD"] == "w/ FD":
                for metric in w_fd_metrics:
                    val = row.get(metric, "N/A")
                    if val != "N/A":
                        w_fd_metrics[metric].append(float(val))
        
        for fd_type, metrics in [("w/o FD", wo_fd_metrics), ("w/ FD", w_fd_metrics)]:
            if metrics["Acc_2"]:
                mean_row = {
                    "Modality": "Mean",
                    "FD": fd_type,
                    "Acc_2": f"{np.mean(metrics['Acc_2']):.4f}",
                    "F1_score": f"{np.mean(metrics['F1_score']):.4f}",
                    "Acc_7": f"{np.mean(metrics['Acc_7']):.4f}",
                    "MAE": f"{np.mean(metrics['MAE']):.4f}",
                }
                std_row = {
                    "Modality": "Std",
                    "FD": fd_type,
                    "Acc_2": f"{np.std(metrics['Acc_2']):.4f}",
                    "F1_score": f"{np.std(metrics['F1_score']):.4f}",
                    "Acc_7": f"{np.std(metrics['Acc_7']):.4f}",
                    "MAE": f"{np.std(metrics['MAE']):.4f}",
                }
                writer.writerow(mean_row)
                writer.writerow(std_row)
    
    print(f"✓ Table4 saved to: {output_path}")


def main():
    table_rows = []
    for modality, variant, fd_tag in VARIANTS:
        result = test_variant(variant)
        if result is None:
            table_rows.append({
                "Modality": modality,
                "FD": fd_tag,
                "Acc_2": "N/A",
                "F1_score": "N/A",
                "Acc_7": "N/A",
                "MAE": "N/A",
            })
            continue
        table_rows.append({
            "Modality": modality,
            "FD": fd_tag,
            "Acc_2": f"{result.get('Acc_2', 0.0):.4f}",
            "F1_score": f"{result.get('F1_score', 0.0):.4f}",
            "Acc_7": f"{result.get('Acc_7', 0.0):.4f}",
            "MAE": f"{result.get('MAE', 0.0):.4f}",
        })

    generate_table4_csv(table_rows)


if __name__ == "__main__":
    main()
