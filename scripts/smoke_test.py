"""
Smoke Test Script for DMD Experiments
Quick validation that all experiments can run and generate unique model files
Runs each experiment for only a few epochs to verify configuration and path isolation
"""
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path
script_dir = Path(__file__).parent
dmd_root = script_dir.parent
sys.path.insert(0, str(dmd_root))

from config import get_config_regression


# Experiment configurations (2 BERT aligned only - matching paper Table 1 & 2)
EXPERIMENTS = [
    {"name": "mosi_aligned_bert", "dataset": "mosi", "config_file": "experiments/configs/mosi_aligned_bert.json"},
    {"name": "mosei_aligned_bert", "dataset": "mosei", "config_file": "experiments/configs/mosei_aligned_bert.json"},
]


def create_smoke_test_config(original_config_path, epochs=2):
    """
    Create a modified config with reduced epochs for smoke testing
    """
    with open(original_config_path, 'r') as f:
        config = json.load(f)
    
    # Reduce early_stop to match the smoke test epochs
    config["dmd"]["commonParams"]["early_stop"] = epochs
    config["dmd"]["commonParams"]["update_epochs"] = 1
    
    return config


def verify_model_files(experiment_results):
    """
    Verify that all experiments generated unique model files
    """
    from pathlib import Path
    print("\n" + "=" * 80)
    print("MODEL FILE VERIFICATION")
    print("=" * 80)
    
    all_paths = []
    duplicates = []
    
    for result in experiment_results:
        exp_name = result["experiment"]
        model_path_str = result["model_path"]
        model_path = Path(model_path_str)
        
        exists = model_path.exists()
        status = "✓ EXISTS" if exists else "✗ MISSING"
        
        print(f"{exp_name:<30} {status:<15} {model_path_str}")
        
        if model_path_str in all_paths:
            duplicates.append(model_path_str)
        else:
            all_paths.append(model_path_str)
    
    print("=" * 80)
    print(f"Total unique paths: {len(set(all_paths))}/{len(experiment_results)}")
    
    if duplicates:
        print(f"⚠ WARNING: Found {len(duplicates)} duplicate paths!")
        for dup in duplicates:
            print(f"  - {dup}")
        return False
    else:
        print("✓ All model paths are unique!")
        return True


def run_smoke_test(epochs=2, cleanup=True):
    """
    Run smoke test for all experiments
    
    Args:
        epochs: Number of epochs to train (default: 2)
        cleanup: Whether to cleanup smoke test files after completion
    """
    print("\n" + "=" * 80)
    print("DMD SMOKE TEST")
    print("=" * 80)
    print(f"Mode: Quick validation ({epochs} epochs per experiment)")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Cleanup after: {'Yes' if cleanup else 'No'}")
    print("=" * 80)
    
    # Create smoke test directory
    smoke_dir = dmd_root / "experiments" / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_results = []
    failed_experiments = []
    
    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp["name"]
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] Testing: {exp_name}")
        print("-" * 80)
        
        # Create experiment-specific smoke test directories
        exp_smoke_dir = smoke_dir / exp_name
        model_dir = exp_smoke_dir / "models"
        log_dir = exp_smoke_dir / "logs"
        
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and modify config
        config_path = dmd_root / exp["config_file"]
        config = create_smoke_test_config(config_path, epochs=epochs)
        
        # Save modified config
        smoke_config_path = exp_smoke_dir / "config.json"
        with open(smoke_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Expected model path (follows pattern: {model_name}-{dataset_name}.pth)
        model_path = model_dir / f"dmd-{exp['dataset']}.pth"
        
        print(f"  Config: {smoke_config_path}")
        print(f"  Model will be saved to: {model_path}")
        
        try:
            # Import and run DMD (with minimal training)
            from run import DMD_run
            
            DMD_run(
                model_name='dmd',
                dataset_name=exp["dataset"],
                config_file=str(smoke_config_path),
                seeds=[1111],
                model_save_dir=str(model_dir),
                res_save_dir=str(exp_smoke_dir / "results"),
                log_dir=str(log_dir),
                mode='train',
                is_distill=True,
                is_tune=False,
                verbose_level=0  # Minimal output for smoke test
            )
            
            # Verify model was created
            if model_path.exists():
                file_size = model_path.stat().st_size / (1024 * 1024)  # MB
                print(f"  ✓ Model file created: {file_size:.2f} MB")
                experiment_results.append({
                    "experiment": exp_name,
                    "model_path": str(model_path),
                    "size_mb": file_size,
                    "status": "success"
                })
            else:
                print(f"  ✗ Model file not found!")
                failed_experiments.append({
                    "experiment": exp_name,
                    "error": "Model file not created"
                })
        
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            failed_experiments.append({
                "experiment": exp_name,
                "error": str(e)
            })
    
    # Print results
    print("\n" + "=" * 80)
    print("SMOKE TEST RESULTS")
    print("=" * 80)
    print(f"Total: {len(EXPERIMENTS)}")
    print(f"Successful: {len(experiment_results)}")
    print(f"Failed: {len(failed_experiments)}")
    print("=" * 80)
    
    # Verify unique paths
    if experiment_results:
        paths_unique = verify_model_files(experiment_results)
    else:
        paths_unique = False
    
    # Print failed experiments
    if failed_experiments:
        print("\n" + "=" * 80)
        print("FAILED EXPERIMENTS")
        print("=" * 80)
        for item in failed_experiments:
            print(f"  ✗ {item['experiment']}: {item['error']}")
    
    # Save results
    results_file = smoke_dir / "smoke_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "epochs": epochs,
            "successful": experiment_results,
            "failed": failed_experiments,
            "paths_unique": paths_unique
        }, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Cleanup
    if cleanup:
        print(f"\nCleaning up smoke test files from {smoke_dir}...")
        try:
            shutil.rmtree(smoke_dir)
            print("✓ Cleanup complete")
        except Exception as e:
            print(f"⚠ Cleanup failed: {e}")
    
    # Final verdict
    print("\n" + "=" * 80)
    if len(experiment_results) == len(EXPERIMENTS) and paths_unique:
        print("✓✓✓ SMOKE TEST PASSED ✓✓✓")
        print("All experiments ran successfully with unique model paths!")
    else:
        print("✗✗✗ SMOKE TEST FAILED ✗✗✗")
        if len(failed_experiments) > 0:
            print(f"  - {len(failed_experiments)} experiment(s) failed")
        if not paths_unique:
            print("  - Model paths are not unique (will overwrite!)")
    print("=" * 80 + "\n")
    
    return len(failed_experiments) == 0 and paths_unique


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smoke test for DMD experiments")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of epochs to train (default: 2)")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Keep smoke test files after completion")
    
    args = parser.parse_args()
    
    success = run_smoke_test(epochs=args.epochs, cleanup=not args.no_cleanup)
    sys.exit(0 if success else 1)
