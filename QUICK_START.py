"""
快速开始指南
复现DMD论文 - 验证主模型性能 (Table 1 & Table 2)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 📋 快速命令速查表
# ═══════════════════════════════════════════════════════════════════════════════

"""
【最简单的方式】- 运行所有8个实验:

    python reproduce_experiments.py --all --gpu 0

【分步运行特定实验】:

    # MOSI数据集 - 对齐 (GloVe特征)
    python reproduce_experiments.py --dataset mosi --aligned --glove

    # MOSI数据集 - 对齐 (BERT特征*)
    python reproduce_experiments.py --dataset mosi --aligned --bert

    # MOSI数据集 - 未对齐
    python reproduce_experiments.py --dataset mosi --unaligned

    # MOSEI数据集 - 对齐 (GloVe特征)
    python reproduce_experiments.py --dataset mosei --aligned --glove

    # MOSEI数据集 - 对齐 (BERT特征*)
    python reproduce_experiments.py --dataset mosei --aligned --bert

    # MOSEI数据集 - 未对齐
    python reproduce_experiments.py --dataset mosei --unaligned

【多GPU加速】(推荐):

    python reproduce_experiments.py --all --gpu 0 1 2

【查看详细指南】:

    python reproduce_experiments.py --guide

【收集并查看结果】:

    python collect_results.py --summary
    python collect_results.py --compare
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 8个实验的详细对应关系
# ═══════════════════════════════════════════════════════════════════════════════

"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ MOSI 实验矩阵 (共3个)                                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ [1] MOSI Aligned (GloVe)        对标: Table 1, Row "DMD (Ours)"         │
│     ✓ need_data_aligned=true                                              │
│     ✓ use_bert=false (300维GloVe特征)                                    │
│     期望ACC7: 41.4%, ACC2: 84.5%, F1: 84.4%                              │
│                                                                              │
│ [2] MOSI Aligned (BERT)*        对标: Table 1, Row "DMD (Ours)*"        │
│     ✓ need_data_aligned=true                                              │
│     ✓ use_bert=true (768维BERT特征)  ← 注意带*号                        │
│     ✓ use_finetune=true (微调BERT)                                      │
│     期望ACC7: 45.6%, ACC2: 86.0%, F1: 86.0%                              │
│                                                                              │
│ [3] MOSI Unaligned              对标: Table 1, Row "DMD (Ours)" Unaligned│
│     ✓ need_data_aligned=false                                             │
│     ✓ use_bert=false (GloVe特征)                                         │
│     期望ACC7: 41.9%, ACC2: 83.5%, F1: 83.5%                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ MOSEI 实验矩阵 (共3个)                                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ [4] MOSEI Aligned (GloVe)       对标: Table 2, Row "DMD (Ours)"         │
│     ✓ need_data_aligned=true                                              │
│     ✓ use_bert=false (300维GloVe特征)                                    │
│     期望ACC7: 53.7%, ACC2: 85.0%, F1: 84.9%                              │
│                                                                              │
│ [5] MOSEI Aligned (BERT)*       对标: Table 2, Row "DMD (Ours)*"        │
│     ✓ need_data_aligned=true                                              │
│     ✓ use_bert=true (768维BERT特征)  ← 注意带*号                        │
│     ✓ use_finetune=true (微调BERT)                                      │
│     期望ACC7: 54.5%, ACC2: 86.6%, F1: 86.6%                              │
│                                                                              │
│ [6] MOSEI Unaligned             对标: Table 2, Row "DMD (Ours)" Unaligned│
│     ✓ need_data_aligned=false                                             │
│     ✓ use_bert=false (GloVe特征)                                         │
│     期望ACC7: 54.6%, ACC2: 84.8%, F1: 84.7%                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ⚙️ 配置参数详解
# ═══════════════════════════════════════════════════════════════════════════════

"""
关键配置参数说明:

┌─ 数据层面 ──────────────────────────────────────────────────────────────────┐
│                                                                              │
│ need_data_aligned: true/false                                             │
│   ├─ true  = 加载 MOSI/Processed/aligned_50.pkl                          │
│   │        时间对齐的特征 (Aligned设置)                                    │
│   └─ false = 加载 MOSI/Processed/unaligned_50.pkl                        │
│            原始未对齐特征 (Unaligned设置)                                  │
│                                                                              │
│ feature_dims:                                                              │
│   ├─ GloVe设置: [300, 74, 35] (MOSI)  或 [300, 74, 35] (MOSEI)         │
│   └─ BERT设置: [768, 74, 35] (MOSI)  或 [768, 74, 35] (MOSEI)         │
│                  ↑ 文本维度从300→768                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 模型层面 ──────────────────────────────────────────────────────────────────┐
│                                                                              │
│ need_model_aligned: true/false                                            │
│   = 模型内部是否加入对齐相关的损失约束 (通常与need_data_aligned相同)       │
│                                                                              │
│ use_bert: true/false                                                       │
│   ├─ true  = 使用 BERT-base-uncased 预训练模型                           │
│   │        （需要下载: /path/to/hugface/bert-base-uncased/）             │
│   └─ false = 使用 GloVe 词向量                                            │
│                                                                              │
│ use_finetune: true/false (仅当use_bert=true时有效)                       │
│   ├─ true  = BERT参数在训练中更新 (微调)                                 │
│   └─ false = 冻结BERT参数,只训练上层模块                                 │
│                                                                              │
│ attn_mask: true/false                                                      │
│   = 处理变长序列时的注意力掩码 (通常保持true)                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 训练层面 ──────────────────────────────────────────────────────────────────┐
│                                                                              │
│ batch_size: 16 (MOSI), 16 (MOSEI)                                        │
│   = 批次大小                                                               │
│                                                                              │
│ learning_rate: 0.0001                                                      │
│   = 学习率                                                                 │
│                                                                              │
│ epochs: 30                                                                 │
│   = 最大训练轮数                                                           │
│                                                                              │
│ early_stop: 10                                                             │
│   = 早停耐心值 (10个epoch无提升则停止)                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 如何判断实验是否成功
# ═══════════════════════════════════════════════════════════════════════════════

"""
成功判断标准:

╔════════════════════════════════════════════════════════════════════════════╗
║ ✓ 完全复现 (推荐)      误差 < 2%                                          ║
║                      说明: 模型配置和超参正确,完全复现了原论文结果     ║
║                                                                          ║
║ ⚠️  部分复现 (可接受)   误差 2% ~ 5%                                      ║
║                      原因: 1. 随机数种子差异                             ║
║                            2. PyTorch版本差异                            ║
║                            3. 硬件(GPU)差异导致数值精度差异               ║
║                      建议: 使用相同的seeds (论文未公开所有seed)           ║
║                            尝试多个seed取平均                            ║
║                                                                          ║
║ ✗ 需要调整 (需排查)    误差 > 5%                                          ║
║                      可能原因: 1. 配置参数设置错误 (最常见)               ║
║                            2. 数据集版本不同                             ║
║                            3. 模型权重加载有问题                         ║
║                            4. 代码bug                                    ║
║                      排查步骤: 检查日志信息 → 检查config参数              ║
║                            → 查看数据加载是否正确                        ║
║                            → 对比代码与论文公开源码                      ║
╚════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 📂 关键文件位置和说明
# ═══════════════════════════════════════════════════════════════════════════════

"""
文件树:

DMD/
 ├─ reproduce_experiments.py      ← 【主脚本】运行实验
 ├─ collect_results.py            ← 【结果脚本】收集结果
 │
 ├─ config/
 │  └─ config.json                ← 配置文件(包含所有参数)
 │
 ├─ result/
 │  ├─ normal/                    ← 实验结果CSV (自动生成)
 │  │  ├─ mosi.csv
 │  │  └─ mosei.csv
 │  └─ experiments/               ← 结果备份 (自动生成)
 │
 ├─ log/
 │  └─ dmd-*.log                  ← 训练日志 (自动生成)
 │
 ├─ pt/
 │  └─ dmd-*.pth                  ← 模型权重 (自动生成)
 │
 ├─ run.py                        ← 核心运行脚本
 ├─ train.py                      ← 训练启动脚本
 ├─ test.py                       ← 测试启动脚本
 ├─ config.py                     ← 配置加载模块
 ├─ data_loader.py                ← 数据加载模块
 └─ trains/                       ← 模型定义
    ├─ singleTask/
    │  └─ model/
    │     └─ dmd.py               ← DMD模型实现
    └─ ...

关键输入:
  dataset/MOSI/Processed/
    ├─ aligned_50.pkl             ← 对齐特征
    └─ unaligned_50.pkl           ← 未对齐特征

  dataset/MOSEI/Processed/
    ├─ aligned_50.pkl
    └─ unaligned_50.pkl
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 完整实验流程
# ═══════════════════════════════════════════════════════════════════════════════

"""
完整实验流程:

第1步: 验证环境和数据
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✓ 检查Python版本及依赖:
    python --version
    pip list | grep torch
    
  ✓ 检查数据集是否已下载:
    ls -la dataset/MOSI/Processed/
    ls -la dataset/MOSEI/Processed/
    
  ✓ 检查预训练模型 (若使用BERT):
    ls -la trains/singleTask/model/hugface/ (或config中配置的路径)

第2步: 运行实验
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  【推荐方式1】一键运行所有8个实验:
  
    python reproduce_experiments.py --all --gpu 0
    
    预期耗时:
      ├─ 单GPU:   ~4-6小时 (MOSI: ~1h, MOSEI: ~3h)
      ├─ 双GPU:   ~2-3小时
      └─ 四GPU:   ~1-1.5小时

  【推荐方式2】分次运行,便于调试:
  
    # 先运行MOSI Aligned (GloVe) - 快速测试
    python reproduce_experiments.py --dataset mosi --aligned --glove --gpu 0
    
    # 检查结果后运行其他
    python reproduce_experiments.py --dataset mosi --aligned --bert --gpu 0
    python reproduce_experiments.py --dataset mosi --unaligned --gpu 0
    
    # 然后运行MOSEI
    python reproduce_experiments.py --dataset mosei --aligned --glove --gpu 0
    python reproduce_experiments.py --dataset mosei --aligned --bert --gpu 0
    python reproduce_experiments.py --dataset mosei --unaligned --gpu 0

第3步: 检查和对比结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✓ 查看实验摘要:
    python collect_results.py --summary
    
  ✓ 与论文期望值对比:
    python collect_results.py --compare
    
  ✓ 导出结果为CSV:
    python collect_results.py --export csv --output my_results.csv

第4步: 结果分析和记录
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  关键文件位置:
    ├─ 詳細日志:     ./log/dmd-*.log
    ├─ 结果表格:     ./result/normal/*.csv
    ├─ 模型权重:     ./pt/dmd-*.pth
    └─ 总结报告:     请自行整理为markdown或pdf

  检查清单:
    ☐ 所有6个实验都已完成
    ☐ 误差都在可接受范围内 (< 5%)
    ☐ 日志中没有ERROR或CRITICAL
    ☐ 保存关键数据和结结果截图
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ⚠️ 常见问题和解决方案
# ═══════════════════════════════════════════════════════════════════════════════

"""
常见问题排查:

Q1: 运行报错 "FileNotFoundError: config/config.json"
A1: 确保在DMD目录下运行脚本:
    cd /Users/mac/Documents/2025tongji/Traoz/UES/DMD/
    python reproduce_experiments.py --all

Q2: 报错 "No module named 'torch' / 'transformers' 等"
A2: 安装依赖 (查看 requirements.txt):
    pip install -r requirements.txt
    
Q3: CUDA out of memory
A3: 减小batch_size或使用更少的GPU:
    python reproduce_experiments.py --all --gpu 0  (只用单卡)
    
    或修改config.json中的batch_size:
    "batch_size": 8  (改小)

Q4: 结果与论文差异太大 (误差 > 5%)
A4: 检查清单:
    ✓ 确认config中aligned/bert参数是否正确
    ✓ 检查数据集是否正确加载 (查看日志中的data shape)
    ✓ 确认是否使用了公开的seed (论文seed可能未公开)
    ✓ 尝试运行同样配置3次,取平均值
    ✓ 对比官方代码 (https://github.com/mdswyz/DMD)

Q5: 需要多GPU加速
A5: 修改命令中的 --gpu 参数:
    python reproduce_experiments.py --all --gpu 0 1 2 3
    
    会自动分配任务到指定GPU (根据脚本实现)

Q6: 只想运行一个数据集的某个配置
A6: 使用命令:
    python reproduce_experiments.py --dataset mosi --unaligned --gpu 0
"""

print(__doc__)  # 打印整个文档
