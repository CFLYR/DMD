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
import logging
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
from datetime import datetime

# 添加DMD项目路径
DMD_DIR = Path(__file__).parent
sys.path.insert(0, str(DMD_DIR))

from run import DMD_run


class ExperimentRunner:
    """实验运行器 - 管理不同的实验配置"""
    
    def __init__(self, config_file="config/config.json", results_dir="result/experiments", log_dir="log"):
        self.config_file = Path(config_file)
        self.results_dir = Path(results_dir)
        self.log_dir = Path(log_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志文件
        self.exp_log_file = self.log_dir / f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._setup_logger()
        self.results_log = []
        
    def _setup_logger(self):
        """配置日志记录器"""
        self.logger = logging.getLogger('DMD_Experiments')
        self.logger.setLevel(logging.DEBUG)
        
        # 文件处理器
        fh = logging.FileHandler(self.exp_log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info("="*80)
        self.logger.info("DMD论文复现 - 实验启动")
        self.logger.info(f"日志文件: {self.exp_log_file}")
        self.logger.info("="*80)
        
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
        start_time = datetime.now()
        
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"【实验开始】{setting_name}")
        self.logger.info("="*80)
        self.logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"数据集: {dataset_name.upper()}")
        self.logger.info(f"对齐模式: {'Aligned' if aligned else 'Unaligned'}")
        self.logger.info(f"特征类型: {'BERT (768dim)' if use_bert else 'GloVe (300dim)'}")
        self.logger.info(f"随机种子: {seeds}")
        self.logger.info(f"GPU列表: {gpu_ids}")
        
        # 准备config修改
        config_updates = {
            'need_data_aligned': aligned,
            'need_model_aligned': aligned,
            'use_bert': use_bert,
            'use_finetune': use_bert,  # 只有用BERT时才微调
        }
        
        # 为此实验生成独特的模型保存路径 (关键!)
        align_str = 'aligned' if aligned else 'unaligned'
        bert_str = 'bert' if use_bert else 'glove'
        model_save_dir = Path('./pt') / dataset_name / align_str / bert_str
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"模型保存目录: {model_save_dir}")
        self.logger.info("-"*80)
        
        try:
            # 运行训练
            self.logger.info("▶ 开始训练...")
            result = DMD_run(
                model_name='dmd',
                dataset_name=dataset_name,
                config=config_updates,
                config_file=str(self.config_file),
                seeds=seeds,
                model_save_dir=str(model_save_dir),
                res_save_dir="./result",
                log_dir="./log",
                mode='train',
                is_distill=True,
                gpu_ids=gpu_ids,
                verbose_level=1
            )
            
            # 运行测试
            self.logger.info("▶ 开始测试...")
            test_result = DMD_run(
                model_name='dmd',
                dataset_name=dataset_name,
                config=config_updates,
                config_file=str(self.config_file),
                seeds=seeds,
                model_save_dir=str(model_save_dir),
                res_save_dir="./result",
                log_dir="./log",
                mode='test',
                is_distill=False,
                gpu_ids=gpu_ids,
                verbose_level=1
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            self.logger.info("-"*80)
            self.logger.info(f"✓ 实验成功完成: {setting_name}")
            self.logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"耗时: {duration:.1f} 分钟")
            self.logger.info("="*80)
            
            # 记录到结果log
            self.results_log.append({
                'setting': setting_name,
                'status': '✓ 通过',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': round(duration, 2),
                'model_dir': str(model_save_dir)
            })
            
            return True
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            self.logger.error("-"*80)
            self.logger.error(f"✗ 实验失败: {setting_name}")
            self.logger.error(f"错误信息: {str(e)}")
            self.logger.error(f"耗时: {duration:.1f} 分钟")
            self.logger.error("="*80)
            
            # 记录到结果log
            self.results_log.append({
                'setting': setting_name,
                'status': '✗ 失败',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': round(duration, 2),
                'error': str(e),
                'model_dir': str(model_save_dir)
            })
            
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
        
        self.logger.info(f"\n【全量实验】开始 - 共{len(experiments)}个配置")
        self.logger.info(f"GPU列表: {gpu_ids}")
        self.logger.info("-"*80)
        
        for i, (dataset, aligned, use_bert, desc) in enumerate(experiments, 1):
            self.logger.info(f"\n[{i}/{len(experiments)}] {desc}")
            success = self.run_experiment(
                dataset_name=dataset,
                aligned=aligned,
                use_bert=use_bert,
                seeds=seeds,
                gpu_ids=gpu_ids
            )
        
        # 保存汇总报告
        self._save_summary_report()
        
        return self.results_log
    
    def _save_summary_report(self):
        """保存实验汇总报告"""
        self.logger.info("\n" + "="*80)
        self.logger.info("【实验汇总报告】")
        self.logger.info("="*80)
        
        passed = sum(1 for r in self.results_log if r['status'] == '✓ 通过')
        failed = sum(1 for r in self.results_log if r['status'] == '✗ 失败')
        total_time = sum(r.get('duration_minutes', 0) for r in self.results_log)
        
        self.logger.info(f"总计: {len(self.results_log)} 个实验")
        self.logger.info(f"成功: {passed} 个 ✓")
        self.logger.info(f"失败: {failed} 个 ✗")
        self.logger.info(f"总耗时: {total_time:.1f} 分钟")
        self.logger.info("-"*80)
        
        for i, result in enumerate(self.results_log, 1):
            status_symbol = "✓" if result['status'] == '✓ 通过' else "✗"
            self.logger.info(
                f"{i}. {status_symbol} {result['setting']:30s} "
                f"({result['duration_minutes']:.1f}min) "
                f"-> {result['model_dir']}"
            )
        
        self.logger.info("="*80)
        self.logger.info(f"完整日志已保存: {self.exp_log_file}\n")
        
        # 同时保存为CSV格式便于后续分析
        csv_file = self.log_dir / f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}_summary.csv"
        df = pd.DataFrame(self.results_log)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"CSV汇总已保存: {csv_file}\n")
    
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
            runner.logger.info(f"\n✓ 实验成功完成!")
            runner.logger.info(f"  日志保存在: {runner.exp_log_file}")
        else:
            runner.logger.error(f"\n✗ 实验执行失败，请检查日志")
            sys.exit(1)


if __name__ == '__main__':
    main()
