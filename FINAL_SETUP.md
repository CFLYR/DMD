# 最终实验设置 - 2个BERT对齐实验

## 精简方案

根据论文，只有以下2个实验有报告的结果（Table 1 & 2中带*号的行）：

| 实验 | 数据集 | 对齐 | 论文预期 ACC7 |
|------|--------|------|------------|
| mosi_aligned_bert | MOSI | ✓ Aligned | **45.6%** |
| mosei_aligned_bert | MOSEI | ✓ Aligned | **54.5%** |

**Unaligned 实验论文中没有报告，所以从脚本中移除。**

---

## 已生成的配置文件

```
experiments/configs/
├── mosi_aligned_bert.json    (use_bert=true, feature_dims=[768, ...])
└── mosei_aligned_bert.json   (use_bert=true, feature_dims=[768, ...])
```

---

## 在服务器上执行

### Step 1: 更新脚本文件

复制或上传以下修改的脚本：
```
DMD/scripts/
├── config_generator.py (✅ 已修改 - 2个实验)
├── smoke_test.py (✅ 已修改 - 2个实验)
├── batch_train.py (✅ 已修改 - 2个实验)
└── batch_test.py (✅ 已修改 - 2个实验)
```

### Step 2: 清理旧的实验文件

```bash
cd /path/to/DMD

# 删除所有GloVe和Unaligned配置
rm -rf experiments/configs/*glove.json
rm -rf experiments/configs/*unaligned*.json
rm -rf experiments/models/*glove
rm -rf experiments/models/*unaligned
rm -rf experiments/smoke_test
```

### Step 3: 重新生成2个配置

```bash
python3 scripts/config_generator.py
```

**预期输出：**
```
DMD Experiment Configuration Generator (BERT Aligned Only)

NOTE: Data files only contain BERT features (768-dim)
      Only aligned experiments from paper tables are reproduced

Generating configurations for 2 experiments...
✓ Generated: experiments/configs/mosi_aligned_bert.json
✓ Generated: experiments/configs/mosei_aligned_bert.json

Configuration Summary
Experiment                Dataset  Aligned    Feature         Expected ACC7
----------------------------------------------------------------------
mosi_aligned_bert         MOSI     Aligned    BERT (768d)     45.6%
mosei_aligned_bert        MOSEI    Aligned    BERT (768d)     54.5%

✓ Successfully generated 2 configuration files
```

### Step 4: 运行Smoke Test（快速验证）

```bash
./run_all.sh --smoke
```

**预期结果：2个实验全部通过**
```
[1/2] Testing: mosi_aligned_bert
  ✓ Model file created: XXX MB

[2/2] Testing: mosei_aligned_bert
  ✓ Model file created: XXX MB

✓ All 2 model paths are unique!
✓ Smoke test completed successfully!
```

**如果通过，继续 Step 5；如果失败，停止并检查日志。**

### Step 5: 完整训练

```bash
./run_all.sh --train
```

会依次训练：
1. **MOSI Aligned BERT** - 预期 ~1-2 小时
2. **MOSEI Aligned BERT** - 预期 ~2-3 小时（更大的数据集）

### Step 6: 评估模型并对比论文结果

```bash
./run_all.sh --test
```

**预期输出：**
```
======================== RESULTS COMPARISON WITH PAPER ========================

Experiment          Dataset  Paper Reference              ACC7 (Our)  ACC7 (Paper)
-----------------------------------------------------------------------------------
mosi_aligned_bert   MOSI     Table 1 - DMD (Ours)*, Aligned  45.6 ✓     45.6
mosei_aligned_bert  MOSEI    Table 2 - DMD (Ours)*, Aligned  54.5 ✓     54.5
```

---

## 生成的模型文件

完成训练后，会在以下位置生成2个模型：

```
experiments/models/
├── mosi_aligned_bert/
│   ├── dmd-mosi.pth      (预期 ~436 MB)
│   └── training.log      (详细训练日志)
├── mosei_aligned_bert/
│   ├── dmd-mosei.pth     (预期 ~425 MB)
│   └── training.log      (详细训练日志)
```

---

## 完整性检查清单

```bash
# 1. 验证配置文件只有2个
ls -lh experiments/configs/
# 输出应该只有 2 个 .json 文件

# 2. 验证config内容
python3 -c "
import json
for f in ['mosi_aligned_bert', 'mosei_aligned_bert']:
    cfg = json.load(open(f'experiments/configs/{f}.json'))
    ds = f.split('_')[0]  # mosi or mosei
    use_bert = cfg['dmd']['commonParams']['use_bert']
    feat_dim = cfg['datasetCommonParams'][ds]['aligned']['feature_dims'][0]
    print(f'{f}: use_bert={use_bert}, text_dim={feat_dim}')
"

# 3. 验证smoke_test.py
grep "EXPERIMENTS = \[" scripts/smoke_test.py -A 5

# 4. 验证batch_train.py
grep "EXPERIMENTS = \[" scripts/batch_train.py -A 5
```

---

## 关键说明

### ✅ 为什么只有2个实验？

1. **数据限制**：'text' 字段存储的是 BERT (768维)，不是 GloVe (300维)
2. **论文限制**：Table 1 和 Table 2 只报告了 Aligned BERT 结果（带 * 号的行）
3. **Unaligned 结果论文中没有报告**，所以没有对比目标

### ✅ 可复现的确切论文结果

- **Table 1**：MOSI Aligned BERT - Expected ACC7: **45.6%**
- **Table 2**：MOSEI Aligned BERT - Expected ACC7: **54.5%**

### ❌ 不可复现的原因

**GloVe 实验：** 数据文件中没有 GloVe (300维) 特征
**Unaligned 实验：** 论文中没有报告结果，无对比目标

---

## 常见问题

**Q: 为什么从4个改成2个？**
A: 论文 Table 1 和 Table 2 只报告了 Aligned 实验的结果。Unaligned 实验论文中没有出现，所以没有预期结果来验证。

**Q: 这 2 个实验足够复现论文吗？**
A: 是的，这是论文的 **主要报告结果**，包含在 Table 1 和 Table 2 中（带*号的行）。

**Q: 为什么GloVe不能跑？**
A: 数据文件的 'text' 字段存的是 BERT (768维)，不是 GloVe (300维)。无法获取 GloVe 特征。

**Q: 可以加上GloVe吗？**
A: 需要从原始文本使用 GloVe 预训练词向量重新提取特征，这是额外的数据预处理工作。当前数据集中没有这些特征。

---

## 日志和结果位置

```
experiments/
├── configs/
│   ├── mosi_aligned_bert.json
│   └── mosei_aligned_bert.json
├── models/
│   ├── mosi_aligned_bert/
│   │   ├── dmd-mosi.pth
│   │   └── training.log
│   └── mosei_aligned_bert/
│       ├── dmd-mosei.pth
│       └── training.log
├── results/
│   ├── comparison_with_paper.csv
│   ├── test_results_detailed.json
│   └── summary.txt
└── smoke_test/          (仅smoke test时生成)
    ├── mosi_aligned_bert/
    └── mosei_aligned_bert/
```

---

## 下一步

运行完整流程：
```bash
# 1. 生成配置
python3 scripts/config_generator.py

# 2. smoke test 验证
./run_all.sh --smoke

# 3. 完整训练（如果smoke test通过）
./run_all.sh --train

# 4. 评估和对比论文结果
./run_all.sh --test
```

预计总耗时：**3-5 小时**（取决于GPU）
