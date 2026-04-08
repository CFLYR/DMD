"""
Main Entry Point for DMD Ablation Study
Wrapper around existing run.py to facilitate ablation experiments

Usage:
  # Train single variant
  python "run will be ablation.py" --mode train --variant variant1_full --dataset mosi
  
  # Train all variants
  python "run will be ablation.py" --mode train --all
  
  # Test all variants and generate Table 3
  python "run will be ablation.py" --mode test
"""
import gc
import logging
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import original DMD_run
from run import DMD_run

# Ablation variant definitions
ABLATION_VARIANTS = [
    'variant1_full',
    'variant2_no_hetero',
    'variant3_no_ca',
    'variant4_only_homo',
    'variant5_only_fd',
    'variant6_baseline'
]

DATASETS = ['mosi', 'mosei']


def train_single_variant(variant, dataset, base_dir=None, epochs=None):
    """Train a single variant
    
    Args:
        variant: Variant name
        dataset: Dataset name
        base_dir: Base directory
        epochs: Override epoch count (for smoke testing)
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    # Use batch_train script
    script_path = base_dir / "scripts" / "batch_train_will be ablation.py"
    
    print("\n" + "=" * 80)
    print(f"Training: {variant} on {dataset.upper()}")
    if epochs:
        print(f"Override Epochs: {epochs}")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        str(script_path),
        "--variant", variant,
        "--dataset", dataset
    ]
    
    if epochs:
        cmd.extend(["--epochs", str(epochs)])
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✓ {variant} on {dataset} completed successfully")
        return True
    else:
        print(f"\n✗ {variant} on {dataset} failed with return code {result.returncode}")
        return False


def train_all_variants(base_dir=None, epochs=None):
    """Train all 12 variants (6 variants × 2 datasets)
    
    Args:
        base_dir: Base directory
        epochs: Override epoch count
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    print("\n" + "=" * 80)
    print("DMD Ablation Study - Training All Variants")
    print("=" * 80)
    print(f"Total experiments: {len(ABLATION_VARIANTS) * len(DATASETS)}")
    if epochs:
        print(f"Override Epochs: {epochs}")
    print("=" * 80)
    
    success_count = 0
    failed = []
    
    for variant in ABLATION_VARIANTS:
        for dataset in DATASETS:
            success = train_single_variant(variant, dataset, base_dir, epochs)
            if success:
                success_count += 1
            else:
                failed.append(f"{variant}_{dataset}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Successful: {success_count}/{len(ABLATION_VARIANTS) * len(DATASETS)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("=" * 80)


def test_all_variants(base_dir=None):
    """Test all variants and generate Table 3"""
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    # Use batch_test script
    script_path = base_dir / "scripts" / "batch_test will be ablation.py"
    
    print("\n" + "=" * 80)
    print("Testing All Variants and Generating Table 3")
    print("=" * 80)
    
    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n✓ Testing completed successfully")
        print("✓ Check experiments/ablation_study/table3_results.csv for results")
    else:
        print(f"\n✗ Testing failed with return code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description='DMD Ablation Study Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single variant (30 epochs default)
  python "run will be ablation.py" --mode train --variant variant1_full --dataset mosi
  
  # Smoke test: Train single variant for 1 epoch
  python "run will be ablation.py" --mode train --variant variant1_full --dataset mosi --epochs 1
  
  # Train all 12 variants
  python "run will be ablation.py" --mode train --all
  
  # Quick smoke test: Train all variants for 1 epoch
  python "run will be ablation.py" --mode train --all --epochs 1
  
  # Test all variants
  python "run will be ablation.py" --mode test
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--variant', type=str,
                       choices=ABLATION_VARIANTS,
                       help='Variant to train (required if not --all)')
    parser.add_argument('--dataset', type=str,
                       choices=DATASETS,
                       help='Dataset to use (required if not --all)')
    parser.add_argument('--all', action='store_true',
                       help='Train/test all variants')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epoch count (default: 30). Use --epochs 1 for smoke test')
    
    args = parser.parse_args()
    
    # Validation
    if args.mode == 'train':
        if not args.all and (not args.variant or not args.dataset):
            parser.error("--variant and --dataset are required when not using --all")
        
        if args.all:
            train_all_variants(epochs=args.epochs)
        else:
            train_single_variant(args.variant, args.dataset, epochs=args.epochs)
    
    elif args.mode == 'test':
        test_all_variants()


if __name__ == "__main__":
    # If run without arguments, show help
    if len(sys.argv) == 1:
        print("=" * 80)
        print("DMD Ablation Study - Main Entry Point")
        print("=" * 80)
        print("\nQuick Start:")
        print("  1. Generate configs:")
        print('     python "scripts/config_generator will be ablation.py"')
        print("\n  2. Train all variants:")
        print('     python "run will be ablation.py" --mode train --all')
        print("\n  3. Test and generate Table 3:")
        print('     python "run will be ablation.py" --mode test')
        print("\nFor more options, run with --help")
        print("=" * 80)
        sys.exit(0)
    
    main()


def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def DMD_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=4, verbose_level=1, mode = '', is_distill = False
):
    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    
    if config_file != "":
        config_file = Path(config_file)
    else: # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    

    args = get_config_regression(model_name, dataset_name, config_file)
    args.is_distill = is_distill  # use or not use distill, train use, test not use
    args.mode = mode # train or test
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'regression'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:
        args.update(config)


    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        result = _run(args, num_workers, is_tune)
        model_results.append(result)
    if args.is_distill:
        criterions = list(model_results[0].keys())
        # save result to csv
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model"] + criterions)
        # save results
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")
    
    # Return results for programmatic access
    # For single seed, return the result directly; for multiple seeds, return first result
    return model_results[0] if len(model_results) == 1 else model_results[0]


def _run(args, num_workers=4, is_tune=False, from_sena=False):

    dataloader = MMDataLoader(args, num_workers)
    if args.is_distill:
        print("training for DMD")

        # param of homogeneous graph distillation
        args.gd_size_low = 64  # hidden size of graph distillation
        args.w_losses_low = [1, 10]  # weights for losses: [logit, repr]
        args.metric_low = 'l1'  # distance metric for distillation loss

        # param of heterogeneous graph distillation
        args.gd_size_high = 32  # hidden size of graph distillation
        args.w_losses_high = [1, 10]  # weights for losses: [logit, repr]
        args.metric_high = 'l1'  # distance metric for distillation loss

        to_idx = [0, 1, 2]  # all modalities can be distilled from each other simultaneously
        from_idx = [0, 1, 2]  # all modalities can be distilled from each other simultaneously
        assert len(from_idx) >= 1

        model = []
        model_dmd = getattr(dmd, 'DMD')(args)
        model_distill_homo = getattr(get_distillation_kernel_homo, 'DistillationKernel')(n_classes=1,
                                                                               hidden_size=
                                                                               args.dst_feature_dim_nheads[0],
                                                                               gd_size=args.gd_size_low,
                                                                               to_idx=to_idx, from_idx=from_idx,
                                                                               gd_prior=softmax([0, 0, 1, 0, 1, 0], 0.25),
                                                                               gd_reg=10,
                                                                               w_losses=args.w_losses_low,
                                                                               metric=args.metric_low,
                                                                               alpha=1 / 8,
                                                                               hyp_params=args)

        model_distill_hetero = getattr(get_distillation_kernel, 'DistillationKernel')(n_classes=1,
                                                                                   hidden_size=
                                                                                   args.dst_feature_dim_nheads[0] * 2,
                                                                                   gd_size=args.gd_size_high,
                                                                                   to_idx=to_idx, from_idx=from_idx,
                                                                                   gd_prior=softmax([0, 0, 1, 0, 1, 1], 0.25),
                                                                                   gd_reg=10,
                                                                                   w_losses=args.w_losses_high,
                                                                                   metric=args.metric_high,
                                                                                   alpha=1 / 8,
                                                                                   hyp_params=args)

        model_dmd, model_distill_homo, model_distill_hetero = model_dmd.cuda(), model_distill_homo.cuda(), model_distill_hetero.cuda()

        model = [model_dmd, model_distill_homo, model_distill_hetero]
    else:
        print("testing phase for DMD")
        model = getattr(dmd, 'DMD')(args)
        model = model.cuda()

    trainer = ATIO().getTrain(args)


    if args.mode == 'test':
        model.load_state_dict(torch.load(args['model_save_path']))
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()
        # Only wait for input if in interactive mode (verbose_level >= 2)
        if args.get('verbose_level', 1) >= 2:
            input('[Press Any Key to start another run]')
    else:
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        model[0].load_state_dict(torch.load(args['model_save_path']))

        results = trainer.do_test(model[0], dataloader['test'], mode="TEST")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results