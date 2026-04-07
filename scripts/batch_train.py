"""
Batch Training Script for DMD Experiments
Trains all 6 experiment configurations sequentially
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path to import DMD modules
script_dir = Path(__file__).parent
dmd_root = script_dir.parent
sys.path.insert(0, str(dmd_root))

from run import DMD_run

# Experiment configurations (2 BERT aligned only - matching paper Table 1 & 2)
EXPERIMENTS = [
    {
        "name": "mosi_aligned_bert",
        "dataset": "mosi",
        "config_file": "experiments/configs/mosi_aligned_bert.json",
        "expected_acc7": 45.6,
    },
    {
        "name": "mosei_aligned_bert",
        "dataset": "mosei",
        "config_file": "experiments/configs/mosei_aligned_bert.json",
        "expected_acc7": 54.5,
    },
]


class ExperimentTracker:
    """Track experiment progress and results"""
    
    def __init__(self, experiments, output_dir):
        self.experiments = experiments
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.failed_experiments = []
        self.start_time = None
        
    def start_experiment(self, exp_idx, exp_name):
        """Log the start of an experiment"""
        print("\n" + "=" * 80)
        print(f"Starting Experiment {exp_idx + 1}/{len(self.experiments)}: {exp_name}")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.start_time and exp_idx > 0:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / exp_idx
            remaining = avg_time * (len(self.experiments) - exp_idx)
            print(f"Estimated Remaining Time: {self._format_time(remaining)}")
        print("=" * 80 + "\n")
        
    def finish_experiment(self, exp_name, success, result=None, error=None):
        """Log the completion of an experiment"""
        if success:
            self.results.append({
                "experiment": exp_name,
                "success": True,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            print("\n" + "=" * 80)
            print(f"✓ Experiment '{exp_name}' completed successfully!")
            if result:
                print(f"Results: {result}")
            print("=" * 80 + "\n")
        else:
            self.failed_experiments.append({
                "experiment": exp_name,
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            })
            print("\n" + "=" * 80)
            print(f"✗ Experiment '{exp_name}' failed!")
            print(f"Error: {error}")
            print("=" * 80 + "\n")
    
    def save_results(self):
        """Save all results to JSON file"""
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "successful": self.results,
                "failed": self.failed_experiments,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
        
        if self.failed_experiments:
            failed_file = self.output_dir / "failed_experiments.txt"
            with open(failed_file, 'w') as f:
                f.write("Failed Experiments\n")
                f.write("=" * 80 + "\n\n")
                for item in self.failed_experiments:
                    f.write(f"Experiment: {item['experiment']}\n")
                    f.write(f"Error: {item['error']}\n")
                    f.write(f"Timestamp: {item['timestamp']}\n")
                    f.write("-" * 80 + "\n\n")
            print(f"✗ Failed experiments logged to: {failed_file}")
    
    def print_summary(self):
        """Print final summary"""
        total_time = time.time() - self.start_time
        print("\n" + "=" * 80)
        print("BATCH TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total Experiments: {len(self.experiments)}")
        print(f"Successful: {len(self.results)}")
        print(f"Failed: {len(self.failed_experiments)}")
        print(f"Total Time: {self._format_time(total_time)}")
        print("=" * 80 + "\n")
        
        if self.results:
            print("Successful Experiments:")
            for item in self.results:
                print(f"  ✓ {item['experiment']}")
        
        if self.failed_experiments:
            print("\nFailed Experiments:")
            for item in self.failed_experiments:
                print(f"  ✗ {item['experiment']}: {item['error']}")
    
    @staticmethod
    def _format_time(seconds):
        """Format seconds into human-readable time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


def run_experiment(exp_config, exp_name):
    """Run a single experiment"""
    dataset = exp_config["dataset"]
    config_file = dmd_root / exp_config["config_file"]
    
    # Create experiment-specific directories
    model_dir = dmd_root / "experiments" / "models" / exp_name
    log_dir = dmd_root / "experiments" / "logs"
    res_dir = dmd_root / "experiments" / "results"
    
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging for this experiment
    log_file = log_dir / f"{exp_name}.log"
    
    # Add experiment info to log
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Dataset: {dataset.upper()}\n")
        f.write(f"Config: {config_file}\n")
        f.write(f"Model Dir: {model_dir}\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
    
    print(f"Dataset: {dataset.upper()}")
    print(f"Config: {config_file}")
    print(f"Model Directory: {model_dir}")
    print(f"Log File: {log_file}")
    print(f"Expected ACC7: {exp_config['expected_acc7']:.1f}%")
    print()
    
    # Run the training
    try:
        result = DMD_run(
            model_name='dmd',
            dataset_name=dataset,
            config_file=str(config_file),
            seeds=[1111],  # Fixed seed as requested
            model_save_dir=str(model_dir),
            res_save_dir=str(res_dir),
            log_dir=str(log_dir),
            mode='train',
            is_distill=True,
            is_tune=False,
            verbose_level=1
        )
        
        # Verify model was saved
        expected_model_path = model_dir / f"dmd-{dataset}.pth"
        if expected_model_path.exists():
            print(f"✓ Model saved to: {expected_model_path}")
        else:
            print(f"⚠ Warning: Expected model at {expected_model_path} not found")
        
        return True, result
    except Exception as e:
        return False, str(e)


def batch_train(experiments=None, continue_from=None):
    """
    Run batch training for all experiments
    
    Args:
        experiments: List of experiment configs (default: all 6 experiments)
        continue_from: Continue from a specific experiment index (0-based)
    """
    if experiments is None:
        experiments = EXPERIMENTS
    
    start_idx = continue_from if continue_from is not None else 0
    
    print("\n" + "=" * 80)
    print("DMD BATCH TRAINING")
    print("=" * 80)
    print(f"Total Experiments: {len(experiments)}")
    if start_idx > 0:
        print(f"Continuing from experiment {start_idx + 1}")
    print(f"Fixed Seed: 1111")
    print("=" * 80)
    
    tracker = ExperimentTracker(experiments, dmd_root / "experiments" / "results")
    tracker.start_time = time.time()
    
    for i, exp in enumerate(experiments):
        if i < start_idx:
            print(f"Skipping experiment {i + 1}: {exp['name']}")
            continue
            
        exp_name = exp["name"]
        tracker.start_experiment(i, exp_name)
        
        success, result = run_experiment(exp, exp_name)
        tracker.finish_experiment(exp_name, success, result=result if success else None, 
                                 error=result if not success else None)
        
        # Small delay between experiments to ensure GPU memory is freed
        if i < len(experiments) - 1:
            print("Waiting 5 seconds before next experiment...")
            time.sleep(5)
    
    tracker.print_summary()
    tracker.save_results()
    
    return tracker.results, tracker.failed_experiments


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch training for DMD experiments")
    parser.add_argument("--continue-from", type=int, default=None,
                       help="Continue from experiment index (0-based)")
    parser.add_argument("--experiment", type=str, default=None,
                       help="Run only a specific experiment by name")
    
    args = parser.parse_args()
    
    experiments = EXPERIMENTS
    if args.experiment:
        experiments = [exp for exp in EXPERIMENTS if exp["name"] == args.experiment]
        if not experiments:
            print(f"Error: Experiment '{args.experiment}' not found!")
            sys.exit(1)
    
    batch_train(experiments, continue_from=args.continue_from)
