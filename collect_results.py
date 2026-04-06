"""
实验结果收集和对比脚本
用途: 统计各实验结果, 与论文期望值进行对比

使用方式:
    python collect_results.py --summary          # 显示实验摘要
    python collect_results.py --compare          # 与论文对比分析
    python collect_results.py --export csv       # 导出为CSV
"""

import json
import csv
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import pandas as pd

# 论文中报告的期望值 - 来自 Table 1 & Table 2
EXPECTED_RESULTS = {
    'mosi_aligned_glove': {
        'ACC7': 41.4, 'ACC2': 84.5, 'F1': 84.4,
        'note': 'Table 1 Aligned (GloVe)'
    },
    'mosi_aligned_bert': {
        'ACC7': 45.6, 'ACC2': 86.0, 'F1': 86.0,
        'note': 'Table 1 Aligned (BERT)*'
    },
    'mosi_unaligned': {
        'ACC7': 41.9, 'ACC2': 83.5, 'F1': 83.5,
        'note': 'Table 1 Unaligned'
    },
    'mosei_aligned_glove': {
        'ACC7': 53.7, 'ACC2': 85.0, 'F1': 84.9,
        'note': 'Table 2 Aligned (GloVe)'
    },
    'mosei_aligned_bert': {
        'ACC7': 54.5, 'ACC2': 86.6, 'F1': 86.6,
        'note': 'Table 2 Aligned (BERT)*'
    },
    'mosei_unaligned': {
        'ACC7': 54.6, 'ACC2': 84.8, 'F1': 84.7,
        'note': 'Table 2 Unaligned'
    }
}


class ResultCollector:
    """实验结果收集器"""
    
    def __init__(self, result_dir='result/experiments', log_dir='log'):
        self.result_dir = Path(result_dir)
        self.log_dir = Path(log_dir)
        self.results_log = []
        
    def collect_csv_results(self, csv_file='result/normal/mosei.csv'):
        """从DMD生成的CSV文件中收集结果"""
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"⚠️  找不到结果文件: {csv_file}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"✗ 读取CSV文件失败: {e}")
            return None
    
    def parse_log_results(self, log_file):
        """从日志文件中解析结果"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # 查找关键指标 (这个需要根据实际日志格式调整)
            metrics = {}
            for line in content.split('\n'):
                if 'ACC7' in line or 'Acc7' in line:
                    # 提取ACC7
                    pass
                if 'ACC2' in line or 'Acc2' in line:
                    # 提取ACC2
                    pass
                if 'F1' in line:
                    # 提取F1
                    pass
            return metrics
        except Exception as e:
            print(f"✗ 解析日志失败: {e}")
            return None
    
    def generate_comparison_table(self):
        """生成对比表"""
        print("\n" + "="*110)
        print("📊 DMD实验结果对比 (与论文期望值)")
        print("="*110)
        print(f"{'实验设置':<30} {'指标':<10} {'论文值':<12} {'实验值':<12} {'误差':<10} {'状态':<8}")
        print("-"*110)
        
        for key, expected in EXPECTED_RESULTS.items():
            # 这里需要从CSV或日志中读取实验值
            # 示例代码
            print(f"{expected['note']:<30} {'ACC7':<10} {expected['ACC7']:<12.2f} {'?':<12} {'?':<10} {'⏳':<8}")
            print(f"{'':30} {'ACC2':<10} {expected['ACC2']:<12.2f} {'?':<12} {'?':<10}")
            print(f"{'':30} {'F1':<10} {expected['F1']:<12.2f} {'?':<12} {'?':<10}")
            print("-"*110)
    
    def generate_summary_visual(self):
        """生成可视化摘要"""
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    实验完成 - 期望检查清单                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

MOSI 实验检查:
  ☐ Aligned (GloVe)    ACC7:{EXPECTED_RESULTS['mosi_aligned_glove']['ACC7']:.1f}%  
  ☐ Aligned (BERT)*    ACC7:{EXPECTED_RESULTS['mosi_aligned_bert']['ACC7']:.1f}%  
  ☐ Unaligned          ACC7:{EXPECTED_RESULTS['mosi_unaligned']['ACC7']:.1f}%  

MOSEI 实验检查:
  ☐ Aligned (GloVe)    ACC7:{EXPECTED_RESULTS['mosei_aligned_glove']['ACC7']:.1f}%  
  ☐ Aligned (BERT)*    ACC7:{EXPECTED_RESULTS['mosei_aligned_bert']['ACC7']:.1f}%  
  ☐ Unaligned          ACC7:{EXPECTED_RESULTS['mosei_unaligned']['ACC7']:.1f}%  

成功标准:
  ✓ 误差 < 2% 为完全复现
  ⚠️  误差 2-5% 为部分复现 (检查超参数配置)
  ✗ 误差 > 5% 为需要调整 (检查日志详情)
"""
        print(summary)


def format_comparison_row(setting, metric, expected_val, actual_val=None):
    """格式化对比行"""
    if actual_val is None:
        status = "⏳ 待测"
        error = "-"
        actual_str = "?"
    else:
        error_val = abs(actual_val - expected_val)
        error = f"{error_val:.2f}%"
        actual_str = f"{actual_val:.2f}%"
        
        if error_val < 2:
            status = "✓ 通过"
        elif error_val < 5:
            status = "⚠️  接近"
        else:
            status = "✗ 需调"
    
    return (setting, metric, f"{expected_val:.2f}%", actual_str, error, status)


def export_results_csv(output_file='comparison_results.csv'):
    """导出对比结果为CSV"""
    rows = []
    
    for key, expected in EXPECTED_RESULTS.items():
        for metric in ['ACC7', 'ACC2', 'F1']:
            rows.append({
                'Setting': expected['note'],
                'Metric': metric,
                'Expected': expected.get(metric, '-'),
                'Actual': '?',
                'Error': '-',
                'Status': 'Pending'
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"✓ 结果导出到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='收集和分析DMD实验结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python collect_results.py --summary
  python collect_results.py --compare mosi
  python collect_results.py --export csv
        """
    )
    
    parser.add_argument('--summary', action='store_true',
                       help='显示实验摘要')
    parser.add_argument('--compare', type=str, nargs='?',
                       help='与论文期望值对比 (mosi/mosei/all)')
    parser.add_argument('--export', type=str, choices=['csv', 'json'],
                       help='导出结果')
    parser.add_argument('--output', type=str, default='comparison_results.csv',
                       help='输出文件名')
    
    args = parser.parse_args()
    
    collector = ResultCollector()
    
    if args.summary:
        collector.generate_summary_visual()
    
    if args.compare:
        collector.generate_comparison_table()
    
    if args.export == 'csv':
        export_results_csv(args.output)
    
    if not any([args.summary, args.compare, args.export]):
        # 默认显示摘要
        collector.generate_summary_visual()


if __name__ == '__main__':
    main()
