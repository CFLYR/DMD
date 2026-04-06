"""
复现论文主实验脚本 (Table 1 & Table 2)
目的：验证完整版DMD模型的SOTA性能

使用方式:
    python reproduce_experiments.py --dataset mosi --aligned
    python reproduce_experiments.py --dataset mosei --unaligned
    python reproduce_experiments.py --all  # 跑所有8个实验组合
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# 添加DMD项目路径
DMD_DIR = Path(__file__).parent
sys.path.insert(0, str(DMD_DIR))

from run import DMD_run


class ExperimentRunner:
    """实验运行器 - 管理不同的实验配置"""
    
    def __init__(self, config_file="config/config.json", results_dir="result/experiments"):
        self.config_file = Path(config_file)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_log = []
        
    def run_experiment(self, dataset_name, aligned=True, use_bert=False, 
                      seeds=[1111, 1112, 1113], gpu_ids=[0]):
        """
        运行单个实验
        
        Args:
            dataset_name: 'mosi' 或 'mosei'
            aligned: True=对齐, False=未对齐
            use_bert: True=BERT特征, False=GloVe特征
            seeds: 随机种子列表
            gpu_ids: GPU编号列表
        
        Returns:
            dict: 实验结果 {accuracy, f1, loss}
        """
        
        setting_name = self._get_setting_name(dataset_name, aligned, use_bert)
        print(f"\n{'='*80}")
        print(f"开始实验: {setting_name}")
        print(f"{'='*80}")
        
        # 准备config修改
        config_updates = {
            'need_data_aligned': aligned,
            'need_model_aligned': aligned,
            'use_bert': use_bert,
            'use_finetune': use_bert,  # 只有用BERT时才微调
        }
        
        try:
            # 运行训练
            result = DMD_run(
                model_name='dmd',
                dataset_name=dataset_name,
                config=config_updates,
                config_file=str(self.config_file),
                seeds=seeds,
                model_save_dir="./pt",
                res_save_dir="./result",
                log_dir="./log",
                mode='train',
                is_distill=True,
                gpu_ids=gpu_ids,
                verbose_level=1
            )
            
            # 运行测试
            test_result = DMD_run(
                model_name='dmd',
                dataset_name=dataset_name,
                config=config_updates,
                config_file=str(self.config_file),
                seeds=seeds,
                model_save_dir="./pt",
                res_save_dir="./result",
                log_dir="./log",
                mode='test',
                is_distill=False,
                gpu_ids=gpu_ids,
                verbose_level=1
            )
            
            print(f"✓ 实验完成: {setting_name}")
            return True
            
        except Exception as e:
            print(f"✗ 实验失败: {setting_name}")
            print(f"  错误信息: {str(e)}")
            return False
    
    def run_all_experiments(self, seeds=[1111, 1112, 1113], gpu_ids=[0]):
        """运行所有8个实验组合"""
        
        experiments = [
            # MOSI
            ('mosi', True, False, 'MOSI Aligned (GloVe)'),
            ('mosi', True, True, 'MOSI Aligned (BERT)*'),
            ('mosi', False, False, 'MOSI Unaligned'),
            
            # MOSEI
            ('mosei', True, False, 'MOSEI Aligned (GloVe)'),
            ('mosei', True, True, 'MOSEI Aligned (BERT)*'),
            ('mosei', False, False, 'MOSEI Unaligned'),
        ]
        
        summary = []
        for dataset, aligned, use_bert, desc in experiments:
            success = self.run_experiment(
                dataset_name=dataset,
                aligned=aligned,
                use_bert=use_bert,
                seeds=seeds,
                gpu_ids=gpu_ids
            )
            
            summary.append({
                'setting': desc,
                'status': '✓ 通过' if success else '✗ 失败'
            })
        
        return summary
    
    def _get_setting_name(self, dataset, aligned, use_bert):
        """生成配置名称"""
        align_str = 'Aligned' if aligned else 'Unaligned'
        feat_str = 'BERT' if use_bert else 'GloVe'
        return f"{dataset.upper()} {align_str} ({feat_str})"


def print_experiment_guide():
    """打印实验指南"""
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    DMD论文复现 - 主模型性能验证                                 ║
║                         (对应 Table 1 & Table 2)                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

📊 需要复现的实验组合 (8个总配置):

┌─ CMU-MOSI ─────────────────────────────────────────────────────────────────────┐
│  [1] Aligned (GloVe)   → 对标 Table 1 Aligned 行                              │
│  [2] Aligned (BERT)*   → 对标 Table 1 Aligned 行(带*)                         │
│  [3] Unaligned         → 对标 Table 1 Unaligned 行                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─ CMU-MOSEI ────────────────────────────────────────────────────────────────────┐
│  [4] Aligned (GloVe)   → 对标 Table 2 Aligned 行                              │
│  [5] Aligned (BERT)*   → 对标 Table 2 Aligned 行(带*)                         │
│  [6] Unaligned         → 对标 Table 2 Unaligned 行                            │
└─────────────────────────────────────────────────────────────────────────────────┘

📋 预期结果对照表:

┌─ MOSI 期待值 ───────────────────────────────────────────┐
│ Aligned (GloVe):  ACC7=41.4%,  ACC2=84.5%,  F1=84.4%   │
│ Aligned (BERT)*:  ACC7=45.6%,  ACC2=86.0%,  F1=86.0%   │
│ Unaligned:        ACC7=41.9%,  ACC2=83.5%,  F1=83.5%   │
└──────────────────────────────────────────────────────────┘

┌─ MOSEI 期待值 ──────────────────────────────────────────┐
│ Aligned (GloVe):  ACC7=53.7%,  ACC2=85.0%,  F1=84.9%   │
│ Aligned (BERT)*:  ACC7=54.5%,  ACC2=86.6%,  F1=86.6%   │
│ Unaligned:        ACC7=54.6%,  ACC2=84.8%,  F1=84.7%   │
└──────────────────────────────────────────────────────────┘

🔧 关键配置参数说明:

  need_data_aligned=True/False   → 数据是否时间对齐加载
  use_bert=True/False            → 文本特征是BERT(768dim)还是GloVe(300dim)
  use_finetune=True/False        → BERT是否微调(仅当use_bert=True时)
  
⚠️  重要注意事项:

  1. 第一次运行需要下载预训练模型 (BERT会较慢)
  2. 多GPU运行: GPU_IDS=[0,1,2] 会加速训练
  3. 结果保存在 ./result/experiments/ 目录
  4. 日志记录在 ./log/ 目录
    """)


def main():
    parser = argparse.ArgumentParser(
        description='DMD论文主实验复现脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行所有实验
  python reproduce_experiments.py --all
  
  # 运行单个实验
  python reproduce_experiments.py --dataset mosi --aligned --glove
  python reproduce_experiments.py --dataset mosei --aligned --bert
  python reproduce_experiments.py --dataset mosi --unaligned
  
  # 指定GPU
  python reproduce_experiments.py --all --gpu 0 1 2
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='运行所有实验组合')
    parser.add_argument('--dataset', type=str, choices=['mosi', 'mosei'],
                       help='数据集选择')
    parser.add_argument('--aligned', action='store_true',
                       help='使用对齐数据')
    parser.add_argument('--unaligned', action='store_true',
                       help='使用未对齐数据')
    parser.add_argument('--bert', action='store_true',
                       help='使用BERT文本特征')
    parser.add_argument('--glove', action='store_true',
                       help='使用GloVe文本特征 (默认)')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                       help='GPU编号列表, 默认为[0]')
    parser.add_argument('--seeds', type=int, nargs='+', 
                       default=[1111, 1112, 1113],
                       help='随机种子列表, 默认为[1111, 1112, 1113]')
    parser.add_argument('--guide', action='store_true',
                       help='显示详细的实验指南')
    
    args = parser.parse_args()
    
    # 显示指南
    if args.guide or (not args.all and not args.dataset):
        print_experiment_guide()
        if not args.dataset and not args.all:
            sys.exit(0)
    
    # 初始化运行器
    runner = ExperimentRunner(
        config_file='config/config.json',
        results_dir='result/experiments'
    )
    
    # 运行实验
    if args.all:
        print_experiment_guide()
        print("\n▶ 开始运行所有实验组合...\n")
        summary = runner.run_all_experiments(seeds=args.seeds, gpu_ids=args.gpu)
        
        print("\n" + "="*80)
        print("📊 实验总结")
        print("="*80)
        for item in summary:
            print(f"  {item['setting']:30s} → {item['status']}")
        print("="*80)
        
    else:
        # 单个实验
        if not args.dataset:
            parser.print_help()
            return
        
        # 确定对齐方式
        if args.unaligned:
            aligned = False
        elif args.aligned:
            aligned = True
        else:
            aligned = True  # 默认对齐
        
        # 确定特征类型
        use_bert = args.bert or (not args.glove and aligned)
        
        success = runner.run_experiment(
            dataset_name=args.dataset,
            aligned=aligned,
            use_bert=use_bert,
            seeds=args.seeds,
            gpu_ids=args.gpu
        )
        
        if success:
            print(f"\n✓ 实验成功完成！")
            print(f"  结果保存在: ./result/experiments/")
            print(f"  日志保存在: ./log/")
        else:
            print(f"\n✗ 实验执行失败，请检查日志")
            sys.exit(1)


if __name__ == '__main__':
    main()
