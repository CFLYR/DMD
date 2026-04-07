# ✅ 已修复：只运行 BERT 实验

## 🔍 问题原因

数据文件检查结果显示：**'text' 字段存储的是 768维 BERT 特征，不是 300维 GloVe！**

```
所有 .pkl 文件中：
  text: (N, 50, 768)        ← BERT特征（尽管名为'text'）
  text_bert: (N, 3, 50)     ← 另一种BERT格式
```

## ✅ 修复方案

已将实验从 6个 减少到 **4个 BERT 实验**：

| 实验名称              | 数据集 | 对齐方式   | 论文预期 ACC7 |
|----------------------|--------|-----------|--------------|
| mosi_aligned_bert    | MOSI   | Aligned   | **45.6%**    |
| mosi_unaligned_bert  | MOSI   | Unaligned | N/A          |
| mosei_aligned_bert   | MOSEI  | Aligned   | **54.5%**    |
| mosei_unaligned_bert | MOSEI  | Unaligned | N/A          |

## 📝 已修改的文件

1. **scripts/config_generator.py** 
   - 移除所有 GloVe 实验配置
   - 只生成 4 个 BERT 配置
   - 全部使用 `use_bert: true`, `feature_dims: [768, ...]`

2. **文档更新**
   - GLOVE_ISSUE.md - 问题分析和解决方案
   - DATA_ANALYSIS.md - 数据文件详细分析报告

## 🚀 在服务器上重新运行

### 1. 拉取最新代码

```bash
cd /path/to/DMD
git pull  # 或者重新上传修改后的文件
```

### 2. 重新生成配置（清理旧的）

```bash
rm -rf experiments/configs/*glove.json  # 删除旧的GloVe配置
python3 scripts/config_generator.py     # 生成4个BERT配置
```

输出应该显示：
```
NOTE: Data files only contain BERT features (768-dim)
      GloVe experiments cannot be reproduced with provided data

Generating configurations for 4 experiments...
✓ Generated: mosi_aligned_bert.json
✓ Generated: mosi_unaligned_bert.json
✓ Generated: mosei_aligned_bert.json
✓ Generated: mosei_unaligned_bert.json
```

### 3. 运行 Smoke Test（快速验证）

```bash
./run_all.sh smoke
```

**预期结果：全部 4 个实验通过**
- ✓ mosi_aligned_bert - 模型文件生成
- ✓ mosi_unaligned_bert - 模型文件生成
- ✓ mosei_aligned_bert - 模型文件生成
- ✓ mosei_unaligned_bert - 模型文件生成

### 4. 完整训练（如果smoke test通过）

```bash
./run_all.sh train
```

训练完成后会在 `experiments/models/` 下生成 4 个模型目录：
```
experiments/models/
├── mosi_aligned_bert/
│   └── dmd-mosi.pth
├── mosi_unaligned_bert/
│   └── dmd-mosi.pth
├── mosei_aligned_bert/
│   └── dmd-mosei.pth
└── mosei_unaligned_bert/
    └── dmd-mosei.pth
```

### 5. 测试（评估模型）

```bash
./run_all.sh test
```

会输出性能对比表格，包含：
- 实际 ACC7 vs 论文预期 ACC7（aligned实验）
- F1, MAE, Corr 等指标

## 📊 可复现的论文结果

**Table 1 - CMU-MOSI:**
- ✅ DMD (Ours)* - Aligned BERT - Expected ACC7: **45.6%**

**Table 2 - CMU-MOSEI:**
- ✅ DMD (Ours)* - Aligned BERT - Expected ACC7: **54.5%**

**不可复现的结果（需要 GloVe 数据）：**
- ❌ MOSI Aligned GloVe (41.4%)
- ❌ MOSEI Aligned GloVe (53.7%)

## 📁 相关文档

- **DATA_ANALYSIS.md** - 数据文件完整分析报告
- **GLOVE_ISSUE.md** - 问题诊断和解决过程
- **CODE_REVIEW.md** - 修复的原仓库bug说明
- **experiments/README.md** - 使用指南

## ❓ 常见问题

**Q: 为什么不能复现 GloVe 实验？**
A: 数据文件的 'text' 字段实际存储的是 BERT (768维) 特征，不是 GloVe (300维)。要复现 GloVe 需要自己从原始文本重新提取特征。

**Q: 论文中的 GloVe 结果是怎么得到的？**
A: 可能作者使用了未公开的数据预处理流程，或者从 CMU-Multimodal SDK 的原始数据自己提取了 GloVe 特征。

**Q: 这 4 个 BERT 实验足够复现论文吗？**
A: 可以复现论文 Table 1 和 Table 2 中带 * 的行（BERT结果），这是论文的主要贡献部分。GloVe 只是对比实验。
