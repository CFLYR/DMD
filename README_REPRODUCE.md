# DMD论文复现 - 快速入门指南

> 用于验证完整版DMD模型的主模型性能  
> 对应论文中的 **Table 1** 和 **Table 2**

---

## 📌 一句话总结

**8个实验配置** × **2个数据集** × **3种设置模式** = 完整复现

```bash
# 最简单的方式: 一键运行所有8个实验
python reproduce_experiments.py --all --gpu 0
```

---

## 🎯 需要进行的8个实验

### MOSI数据集 (3个实验)

| # | 配置 | 对齐 | 特征 | 论文期望值 | 脚本命令 |
|----|------|------|------|----------|---------|
| 1 | MOSI Aligned (GloVe) | ✓ | 300dim | ACC7: 41.4% | `--dataset mosi --aligned --glove` |
| 2 | MOSI Aligned (BERT)* | ✓ | 768dim | ACC7: 45.6% | `--dataset mosi --aligned --bert` |
| 3 | MOSI Unaligned | ✗ | 300dim | ACC7: 41.9% | `--dataset mosi --unaligned` |

### MOSEI数据集 (3个实验)

| # | 配置 | 对齐 | 特征 | 论文期望值 | 脚本命令 |
|----|------|------|------|----------|---------|
| 4 | MOSEI Aligned (GloVe) | ✓ | 300dim | ACC7: 53.7% | `--dataset mosei --aligned --glove` |
| 5 | MOSEI Aligned (BERT)* | ✓ | 768dim | ACC7: 54.5% | `--dataset mosei --aligned --bert` |
| 6 | MOSEI Unaligned | ✗ | 300dim | ACC7: 54.6% | `--dataset mosei --unaligned` |

---

## 🚀 快速开始

### 方式1: 一键运行所有实验（推荐）

```bash
cd /Users/mac/Documents/2025tongji/Traoz/UES/DMD/

# 单GPU版本 (耗时4-6小时)
python reproduce_experiments.py --all --gpu 0

# 多GPU加速 (推荐, 耗时1-2小时)
python reproduce_experiments.py --all --gpu 0 1 2
```

### 方式2: 逐个运行实验（推荐用于调试）

```bash
# MOSI - 对齐GloVe (最快, 用于测试)
python reproduce_experiments.py --dataset mosi --aligned --glove

# MOSI - 对齐BERT
python reproduce_experiments.py --dataset mosi --aligned --bert

# MOSI - 未对齐
python reproduce_experiments.py --dataset mosi --unaligned

# MOSEI - 对齐GloVe
python reproduce_experiments.py --dataset mosei --aligned --glove

# MOSEI - 对齐BERT
python reproduce_experiments.py --dataset mosei --aligned --bert

# MOSEI - 未对齐
python reproduce_experiments.py --dataset mosei --unaligned
```

### 查看详细指南

```bash
# 显示完整的实验指南
python reproduce_experiments.py --guide

# 查看快速开始文档
python QUICK_START.py
```

---

## 📊 检查和分析结果

### 查看实验摘要

```bash
python collect_results.py --summary
```

输出示例：
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    实验完成 - 期望检查清单                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

MOSI 实验检查:
  ☐ Aligned (GloVe)    ACC7:41.4%  
  ☐ Aligned (BERT)*    ACC7:45.6%  
  ☐ Unaligned          ACC7:41.9%  

MOSEI 实验检查:
  ☐ Aligned (GloVe)    ACC7:53.7%  
  ☐ Aligned (BERT)*    ACC7:54.5%  
  ☐ Unaligned          ACC7:54.6%  
```

### 与论文对比

```bash
python collect_results.py --compare
```

### 导出结果为CSV

```bash
python collect_results.py --export csv --output my_results.csv
```

---

## ⚙️ 关键配置参数说明

### 数据层 (如何加载数据)

| 参数 | 取值 | 含义 |
|------|------|------|
| `need_data_aligned` | true | 加载对齐特征 (aligned_50.pkl) |
| | false | 加载未对齐特征 (unaligned_50.pkl) |
| `use_bert` | true | 文本特征用BERT (768维) |
| | false | 文本特征用GloVe (300维) |

### 模型层 (模型如何处理)

| 参数 | 取值 | 含义 |
|------|------|------|
| `need_model_aligned` | true | 模型内部加入对齐约束 |
| `use_finetune` | true | BERT参数在训练中更新 (仅当use_bert=true) |
| `attn_mask` | true | 使用注意力掩码处理变长序列 |

### 训练层

| 参数 | MOSI | MOSEI | 含义 |
|------|------|-------|------|
| `batch_size` | 16 | 16 | 批次大小 |
| `learning_rate` | 0.0001 | 0.0001 | 学习率 |
| `epochs` | 30 | 30 | 最大训练轮数 |
| `early_stop` | 10 | 10 | 早停耐心值 |

---

## ✅ 成功判断标准

| 误差范围 | 状态 | 含义 | 行动 |
|---------|------|------|------|
| < 2% | ✓ 完全复现 | 完全复现了原论文结果 | ✓ 通过 |
| 2% ~ 5% | ⚠️ 部分复现 | 存在minor差异 (通常因随机数种子或硬件) | 检查日志 |
| > 5% | ✗ 需要调整 | 需要检查配置或代码 | 排查问题 |

---

## 📂 文件位置和输出

### 输入数据

```
dataset/
├─ MOSI/Processed/
│  ├─ aligned_50.pkl           ← 对齐特征
│  └─ unaligned_50.pkl         ← 未对齐特征
└─ MOSEI/Processed/
   ├─ aligned_50.pkl
   └─ unaligned_50.pkl
```

### 输出位置

```
DMD/
├─ log/
│  └─ dmd-*.log                ← 详细训练日志
│
├─ pt/
│  └─ dmd-*.pth                ← 保存的模型权重
│
├─ result/
│  ├─ normal/
│  │  ├─ mosi.csv              ← MOSI结果
│  │  └─ mosei.csv             ← MOSEI结果
│  └─ experiments/
│     └─ [备份结果]
```

---

## 🔍 如何查看训练过程

### 查看实时日志

```bash
# 在另一个终端窗口实时查看日志
tail -f log/dmd-mosi.log

# 搜索特定指标
grep "ACC7\|ACC2\|F1" log/dmd-*.log
```

### 检查是否训练正常进行

```bash
# 查找成功标记
grep "Results saved to" log/dmd-*.log

# 查找错误
grep "ERROR\|CRITICAL" log/dmd-*.log
```

---

## ⚠️ 常见问题排查

### Q1: 报错 "No module named 'torch'"

**解决办法**：安装依赖
```bash
pip install -r requirements.txt
```

### Q2: "CUDA out of memory"

**解决办法**：
- 使用单GPU: `--gpu 0`
- 或修改 `config.json` 中的 `batch_size` 改为 8

### Q3: 找不到 config.json

**解决办法**：确保在DMD目录下运行
```bash
cd /Users/mac/Documents/2025tongji/Traoz/UES/DMD/
python reproduce_experiments.py --all
```

### Q4: 结果与论文差异 > 5%

**排查清单**：
- ✓ 确认配置参数正确 (aligned/bert)
- ✓ 查看日志中数据shape是否正确
- ✓ 尝试多次运行取平均值
- ✓ 对比官方代码: https://github.com/mdswyz/DMD

### Q5: 想要加速训练

**办法**：使用多GPU
```bash
python reproduce_experiments.py --all --gpu 0 1 2 3
```

---

## 📈 预期的实验耗时

| 配置 | 单GPU | 双GPU | 四GPU |
|------|-------|-------|-------|
| 单个实验 (e.g., MOSI Aligned) | 45-60min | 25-30min | 15-20min |
| 所有6个实验 | 5-6小时 | 2.5-3小时 | 1.5-2小时 |

---

## 📝 重要提示

### ⭐ 必须注意的配置差异

1. **带 `*` 号的行 = 使用BERT特征**
   - 论文中 Table 1 和 Table 2 的某些行后面有 `*`
   - 这表示使用 768维BERT特征，而非300维GloVe
   - 配置: `use_bert=true`

2. **Aligned vs Unaligned**
   - Aligned: 加载时间对齐的特征 (`need_data_aligned=true`)
   - Unaligned: 加载原始未对齐特征 (`need_data_aligned=false`)

3. **不同数据集特征维度可能不同**
   - MOSI: `[300/768, 5, 20]` (GloVe/BERT, Acoustic, Visual)
   - MOSEI: `[300/768, 74, 35]` (GloVe/BERT, Acoustic, Visual)

---

## 🎓 理解实验的核心

本复现脚本的目的是验证：

> **完整版DMD模型是否能够达到论文报告的SOTA(最优)性能**

关键检查点：

```
✓ 模型结构是否正确实现  
  └─ 特征解耦 (Xcom 和 Xprt)
  └─ 图蒸馏单元 (HomoGD 和 HeteroGD)
  └─ 自回归机制和损失函数

✓ 超参数是否与论文一致  
  └─ λ1, λ2, γ 的平衡因子
  └─ 学习率、批次大小等

✓ 数据预处理是否正确  
  └─ 特征提取 (GloVe vs BERT)
  └─ 时间对齐处理

✓ 评估指标是否一致  
  └─ 7-class accuracy (ACC7)
  └─ 2-class binary accuracy (ACC2)  
  └─ F1-score
```

如果所有6个实验的误差都 < 5%，说明复现成功！

---

## 📚 相关资源

- **论文**: Li et al., "Decoupled Multimodal Distilling for Emotion Recognition", CVPR 2023
- **官方代码**: https://github.com/mdswyz/DMD
- **数据集**:
  - [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/datasets/mosei/data/MOSI/)
  - [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/datasets/mosei/data/MOSEI/)

---

## 💡 最佳实践

1. **先运行最简单的配置测试**
   ```bash
   python reproduce_experiments.py --dataset mosi --aligned --glove --gpu 0
   ```
   预期耗时：45-60分钟。快速验证环境是否正常。

2. **逐个运行各配置**
   按照表格顺序逐一运行，便于监控和调试。

3. **多次运行取平均**
   由于随机性，建议对关键配置运行3次并取平均值。

4. **保存关键数据**
   - 截图保存最终结果表
   - 保存日志文件用于问题排查
   - 与论文对比写入总结报告

5. **版本控制**
   记录好运行时的：
   - Python版本
   - PyTorch版本
   - CUDA版本
   - 数据集日期
   便于后续重现或改进

---

## 📞 获取帮助

如遇问题，按以下顺序排查：

1. 查看 [快速开始指南](QUICK_START.py) 中的常见问题
2. 查看训练日志: `log/dmd-*.log`
3. 参考官方代码: https://github.com/mdswyz/DMD
4. 检查论文的补充材料(supplementary material)

---

**祝实验顺利！** 🎉
