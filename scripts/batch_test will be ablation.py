"""
Batch Testing Script for DMD Ablation Study
Tests all trained ablation variants and generates Table 3 comparison
"""
import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

# Add parent directory to path
script_dir = Path(__file__).parent
dmd_root = script_dir.parent
sys.path.insert(0, str(dmd_root))

from run import DMD_run


# Ablation variant definitions (matching Table 3 in paper)
ABLATION_VARIANTS = [
    {"name": "variant1_full", "description": "Full Model (All Components)"},
    {"name": "variant2_no_hetero", "description": "w/o HeteroGD"},
    {"name": "variant3_no_ca", "description": "Keep Distillation but w/o CA"},
    {"name": "variant4_only_homo", "description": "Only HomoGD"},
    {"name": "variant5_only_fd", "description": "Only FD"},
    {"name": "variant6_baseline", "description": "Baseline (No Advanced Modules)"},
]

DATASETS = ["mosi", "mosei"]


def test_variant(variant_name, dataset_name, config_dir="experiments/ablation_study/configs",
                 model_dir="experiments/ablation_study/models", results_dir="experiments/ablation_study/results"):
    """
    Test a single ablation variant
    
    Returns:
        dict: Test results with ACC7, ACC2, F1, Loss metrics
    """
    # Construct paths
    config_file = dmd_root / config_dir / f"{variant_name}_{dataset_name}.json"
    model_path = dmd_root / model_dir / f"{variant_name}_{dataset_name}" / f"dmd-{dataset_name}.pth"
    results_save_dir = dmd_root / results_dir
    results_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTesting: {variant_name} on {dataset_name.upper()}")
    print("-" * 80)
    print(f"  Model: {model_path}")
    
    # Check if model and config exist
    if not model_path.exists():
        print(f"  ✗ Model not found! Skipping...")
        return None
    if not config_file.exists():
        print(f"  ✗ Config not found! Skipping...")
        return None
    
    # Check model size
    model_size = model_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  Model Size: {model_size:.2f} MB")
    
    # Run testing
    try:
        results = DMD_run(
            model_name='dmd',
            dataset_name=dataset_name,
            config_file=str(config_file),
            seeds=[1111],  # Fixed seed
            model_save_dir=str(model_path.parent),
            res_save_dir=str(results_save_dir),
            log_dir=str(results_save_dir.parent / "logs"),
            mode='test',
            is_distill=True
        )
        
        # Extract metrics (assuming DMD_run returns results dict)
        # Format: {'ACC7': x, 'ACC2': y, 'F1': z, 'Loss': w}
        print(f"  ✓ Test completed successfully!")
        if results:
            print(f"    ACC7: {results.get('ACC7', 'N/A')}")
            print(f"    ACC2: {results.get('ACC2', 'N/A')}")
            print(f"    F1: {results.get('F1', 'N/A')}")
        
        return results
        
    except Exception as e:
        print(f"  ✗ Testing failed: {str(e)}")
        return None


def generate_table3_csv(all_results, output_file="experiments/ablation_study/table3_results.csv"):
    """
    Generate Table 3 comparison CSV
    
    Format:
    Variant | MOSI_ACC7 | MOSI_ACC2 | MOSI_F1 | MOSEI_ACC7 | MOSEI_ACC2 | MOSEI_F1
    """
    output_path = dmd_root / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Generating Table 3 Comparison")
    print("=" * 80)
    
    # Prepare rows
    rows = []
    for variant_info in ABLATION_VARIANTS:
        variant_name = variant_info["name"]
        variant_desc = variant_info["description"]
        
        row = {"Variant": variant_desc}
        
        for dataset in DATASETS:
            key = f"{variant_name}_{dataset}"
            if key in all_results and all_results[key]:
                res = all_results[key]
                row[f"{dataset.upper()}_ACC7"] = f"{res.get('ACC7', 0.0):.2f}"
                row[f"{dataset.upper()}_ACC2"] = f"{res.get('ACC2', 0.0):.2f}"
                row[f"{dataset.upper()}_F1"] = f"{res.get('F1', 0.0):.2f}"
            else:
                row[f"{dataset.upper()}_ACC7"] = "N/A"
                row[f"{dataset.upper()}_ACC2"] = "N/A"
                row[f"{dataset.upper()}_F1"] = "N/A"
        
        rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        fieldnames = ["Variant", "MOSI_ACC7", "MOSI_ACC2", "MOSI_F1", 
                      "MOSEI_ACC7", "MOSEI_ACC2", "MOSEI_F1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ Table 3 saved to: {output_path}")
    
    # Also print to console
    print("\nTable 3 - Ablation Study Results:")
    print("-" * 120)
    print(f"{'Variant':<40} {'MOSI ACC7':>10} {'MOSI F1':>10} {'MOSEI ACC7':>10} {'MOSEI F1':>10}")
    print("-" * 120)
    for row in rows:
        print(f"{row['Variant']:<40} {row['MOSI_ACC7']:>10} {row['MOSI_F1']:>10} "
              f"{row['MOSEI_ACC7']:>10} {row['MOSEI_F1']:>10}")
    print("-" * 120)


def main():
    """Test all ablation variants and generate comparison table"""
    print("=" * 80)
    print("DMD Ablation Study - Batch Testing")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    all_results = {}
    
    # Test all variants
    for variant_info in ABLATION_VARIANTS:
        variant_name = variant_info["name"]
        for dataset in DATASETS:
            key = f"{variant_name}_{dataset}"
            result = test_variant(variant_name, dataset)
            if result:
                all_results[key] = result
    
    # Generate comparison table
    generate_table3_csv(all_results)
    
    print("\n" + "=" * 80)
    print("✓ Batch testing completed!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
    print(f"  Model size: {model_size:.2f} MB")
    
    try:
        # Run testing
        result = DMD_run(
            model_name='dmd',
            dataset_name=dataset,
            config_file=str(config_file),
            seeds=[1111],
            model_save_dir=str(model_dir),
            res_save_dir=str(dmd_root / "experiments" / "results"),
            log_dir=str(dmd_root / "experiments" / "logs"),
            mode='test',
            is_distill=False,  # Testing mode doesn't need distillation
            is_tune=False,
            verbose_level=1
        )
        
        print(f"  ✓ Testing complete")
        return result, None
    
    except Exception as e:
        print(f"  ✗ Testing failed: {str(e)}")
        return None, str(e)


def format_metric(value, expected=None):
    """Format metric with comparison to expected value"""
    if value is None:
        return "N/A"
    
    formatted = f"{value:.1f}"
    if expected is not None:
        diff = value - expected
        if abs(diff) < 0.5:
            formatted += " ✓"
        elif diff > 0:
            formatted += f" (+{diff:.1f})"
        else:
            formatted += f" ({diff:.1f})"
    
    return formatted


def batch_test():
    """Run batch testing for all trained models"""
    print("\n" + "=" * 80)
    print("DMD BATCH TESTING")
    print("=" * 80)
    print(f"Total Experiments: {len(EXPERIMENTS)}")
    print("=" * 80)
    
    results = []
    failed = []
    
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}]", end=" ")
        result, error = test_single_experiment(exp)
        
        if result is not None:
            # Extract metrics (result is a dict with keys: Acc_2, F1_score, Acc_7, MAE, Loss)
            results.append({
                "experiment": exp["name"],
                "dataset": exp["dataset"].upper(),
                "paper_ref": exp["paper_ref"],
                "result": result,
                "expected": exp["expected"]
            })
        else:
            failed.append({
                "experiment": exp["name"],
                "error": error
            })
    
    # Generate comparison table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON WITH PAPER")
    print("=" * 80)
    
    if results:
        # Create DataFrame for better formatting
        table_data = []
        for item in results:
            exp_name = item["experiment"]
            result = item["result"]
            expected = item["expected"]
            
            # Extract metrics from result
            # DMD returns: Acc_2, F1_score, Acc_7, MAE, Loss (all in 0-1 range except MAE)
            row = {
                "Experiment": exp_name,
                "Dataset": item["dataset"],
                "Paper Reference": item["paper_ref"],
            }
            
            # Map result keys to paper metric names and convert to percentage
            metric_mapping = {
                "ACC7": "Acc_7",
                "ACC2": "Acc_2", 
                "F1": "F1_score"
            }
            
            for paper_metric, result_key in metric_mapping.items():
                result_value = None
                if result_key in result:
                    # Convert from 0-1 range to percentage
                    result_value = result[result_key] * 100
                
                expected_value = expected.get(paper_metric)
                row[f"{paper_metric} (Our)"] = format_metric(result_value, expected_value)
                row[f"{paper_metric} (Paper)"] = f"{expected_value:.1f}" if expected_value else "N/A"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_file = dmd_root / "experiments" / "results" / "comparison_with_paper.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Results saved to: {csv_file}")
        
        # Save detailed JSON
        json_file = dmd_root / "experiments" / "results" / "test_results_detailed.json"
        with open(json_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "failed": failed
            }, f, indent=2)
        print(f"✓ Detailed results saved to: {json_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)
    print(f"Total: {len(EXPERIMENTS)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed Experiments:")
        for item in failed:
            print(f"  ✗ {item['experiment']}: {item['error']}")
    
    print("=" * 80 + "\n")
    
    return results, failed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch testing for DMD experiments")
    parser.add_argument("--experiment", type=str, default=None,
                       help="Test only a specific experiment by name")
    
    args = parser.parse_args()
    
    experiments = EXPERIMENTS
    if args.experiment:
        experiments = [exp for exp in EXPERIMENTS if exp["name"] == args.experiment]
        if not experiments:
            print(f"Error: Experiment '{args.experiment}' not found!")
            sys.exit(1)
        # Update global EXPERIMENTS for batch_test function
        EXPERIMENTS = experiments
    
    results, failed = batch_test()
    sys.exit(0 if len(failed) == 0 else 1)
