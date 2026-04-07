"""
Batch Testing Script for DMD Experiments
Tests all trained models and generates comparison table with paper results
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
script_dir = Path(__file__).parent
dmd_root = script_dir.parent
sys.path.insert(0, str(dmd_root))

from run import DMD_run

# Experiment configurations (2 BERT aligned only - exactly matching paper Table 1 & 2)
EXPERIMENTS = [
    {
        "name": "mosi_aligned_bert",
        "dataset": "mosi",
        "config_file": "experiments/configs/mosi_aligned_bert.json",
        "model_dir": "experiments/models/mosi_aligned_bert",
        "expected": {"ACC7": 45.6, "ACC2": 86.0, "F1": 86.0},
        "paper_ref": "Table 1 - DMD (Ours)*, Aligned"
    },
    {
        "name": "mosei_aligned_bert",
        "dataset": "mosei",
        "config_file": "experiments/configs/mosei_aligned_bert.json",
        "model_dir": "experiments/models/mosei_aligned_bert",
        "expected": {"ACC7": 54.5, "ACC2": 86.6, "F1": 86.6},
        "paper_ref": "Table 2 - DMD (Ours)*, Aligned"
    },
]


def test_single_experiment(exp_config):
    """Test a single trained model"""
    exp_name = exp_config["name"]
    dataset = exp_config["dataset"]
    config_file = dmd_root / exp_config["config_file"]
    model_dir = dmd_root / exp_config["model_dir"]
    
    # Model filename follows the pattern: {model_name}-{dataset_name}.pth
    model_path = model_dir / f"dmd-{dataset}.pth"
    
    print(f"\nTesting: {exp_name}")
    print("-" * 80)
    print(f"  Model: {model_path}")
    
    # Check if model exists
    if not model_path.exists():
        print(f"  ✗ Model not found!")
        return None, "Model not found"
    
    # Check model size
    model_size = model_path.stat().st_size / (1024 * 1024)  # MB
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
        
        if result:
            # Extract metrics (assuming result is a dict with metric keys)
            # The actual keys depend on the DMD_run return format
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
            
            # Try to extract metrics from result
            # Note: The actual metric keys depend on DMD implementation
            # Common keys might be: 'Mult_acc_7', 'Mult_acc_2', 'Mult_F1_score'
            row = {
                "Experiment": exp_name,
                "Dataset": item["dataset"],
                "Paper Reference": item["paper_ref"],
            }
            
            # Add metrics with comparison to paper
            for metric in ["ACC7", "ACC2", "F1"]:
                # Try different possible key formats
                result_value = None
                for key in [metric, f"Mult_acc_7", f"Mult_acc_2", f"Mult_F1_score", 
                           f"{metric.lower()}", f"test_{metric.lower()}"]:
                    if key in result:
                        result_value = result[key] * 100 if result[key] <= 1 else result[key]
                        break
                
                expected_value = expected.get(metric)
                row[f"{metric} (Our)"] = format_metric(result_value)
                row[f"{metric} (Paper)"] = f"{expected_value:.1f}" if expected_value else "N/A"
            
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
