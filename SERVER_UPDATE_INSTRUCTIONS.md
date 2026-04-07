# 服务器更新说明 - BERT-Only 实验

## 问题修复

已修复脚本不一致问题：
- ✅ config_generator.py - 只生成4个BERT配置
- ✅ smoke_test.py - 运行4个BERT实验
- ✅ batch_train.py - 训练4个BERT实验  
- ✅ batch_test.py - 测试4个BERT实验

## 在服务器上执行

### Step 1: 重新拉取代码或上传更新的文件

```bash
cd /path/to/DMD

# 如果使用git：
git pull

# 或者手动复制新文件：
# - scripts/config_generator.py (更新)
# - scripts/smoke_test.py (更新)
# - scripts/batch_train.py (更新)
# - scripts/batch_test.py (更新)
```

### Step 2: 清理旧的GloVe配置和模型

```bash
# 删除旧的GloVe配置文件
rm -rf experiments/configs/*glove.json

# 删除旧的GloVe模型目录（可选，节省空间）
rm -rf experiments/models/*glove

# 删除旧的smoke test目录
rm -rf experiments/smoke_test
```

### Step 3: 重新生成配置（4个BERT）

```bash
python3 scripts/config_generator.py
```

**预期输出：**
```
DMD Experiment Configuration Generator (BERT Only)

NOTE: Data files only contain BERT features (768-dim)
      GloVe experiments cannot be reproduced with provided data

Generating configurations for 4 experiments...
✓ Generated: experiments/configs/mosi_aligned_bert.json
✓ Generated: experiments/configs/mosi_unaligned_bert.json
✓ Generated: experiments/configs/mosei_aligned_bert.json
✓ Generated: experiments/configs/mosei_unaligned_bert.json

Configuration Summary
Experiment                Dataset  Aligned    Feature         Expected ACC7
----------------------------------------------------------------------
mosi_aligned_bert         MOSI     Aligned    BERT (768d)     45.6%
mosi_unaligned_bert       MOSI     Unaligned  BERT (768d)     N/A
mosei_aligned_bert        MOSEI    Aligned    BERT (768d)     54.5%
mosei_unaligned_bert      MOSEI    Unaligned  BERT (768d)     N/A

✓ Successfully generated 4 configuration files
```

### Step 4: 运行Smoke Test（快速验证）

```bash
./run_all.sh --smoke
```

**预期结果：全部4个实验通过**
```
[1/4] Testing: mosi_aligned_bert
  ✓ Model file created: XXX MB

[2/4] Testing: mosi_unaligned_bert
  ✓ Model file created: XXX MB

[3/4] Testing: mosei_aligned_bert
  ✓ Model file created: XXX MB

[4/4] Testing: mosei_unaligned_bert
  ✓ Model file created: XXX MB

✓ All 4 model paths are unique!
✓ Smoke test completed successfully!
```

### Step 5: 完整训练（如果smoke test通过）

```bash
./run_all.sh --train
```

### Step 6: 评估模型

```bash
./run_all.sh --test
```

会输出结果对比表格

---

## 文件更新清单

### 必须更新的脚本

```
DMD/scripts/
├── config_generator.py (✅ 更新 - 4个BERT实验)
├── smoke_test.py (✅ 更新 - 4个BERT实验)
├── batch_train.py (✅ 更新 - 4个BERT实验)
└── batch_test.py (✅ 更新 - 4个BERT实验)
```

### 文档文件（参考）

```
DMD/
├── BERT_ONLY_UPDATE.md (新增 - 本更新说明)
├── DATA_ANALYSIS.md (新增 - 数据详细分析)
├── GLOVE_ISSUE.md (更新 - 问题诊断)
└── experiments/
    └── configs/
        ├── mosi_aligned_bert.json (新增)
        ├── mosi_unaligned_bert.json (新增)
        ├── mosei_aligned_bert.json (新增)
        └── mosei_unaligned_bert.json (新增)
```

---

## 验证清单

运行以下命令验证更新是否正确：

```bash
# 1. 检查生成的配置文件
ls -lh experiments/configs/
# 应该只有4个.json文件（*bert.json）

# 2. 验证配置内容
python3 -c "
import json
for f in ['mosi_aligned_bert', 'mosei_aligned_bert']:
    cfg = json.load(open(f'experiments/configs/{f}.json'))
    print(f'{f}: use_bert={cfg[\"dmd\"][\"commonParams\"][\"use_bert\"]}')"

# 3. 检查smoke test会使用的文件
grep "EXPERIMENTS = \[" scripts/smoke_test.py -A 10

# 4. 检查batch_train会使用的文件
grep "EXPERIMENTS = \[" scripts/batch_train.py -A 10
```

---

## 如果出现错误

### 错误 1: FileNotFoundError: config not found

**原因：** 还有旧的GloVe配置文件引用

**解决：**
```bash
grep -r "glove" scripts/*.py  # 查看是否还有GloVe引用
```

### 错误 2: 维度不匹配

**原因：** config.json 使用了 GloVe (300-dim) 而数据是 BERT (768-dim)

**解决：**
```bash
# 重新生成配置
rm experiments/configs/*.json
python3 scripts/config_generator.py
```

### 错误 3: 模型文件覆盖

**原因：** 两个实验使用了相同的模型保存路径

**解决：** 这不应该发生，因为已经修复了DMD.py。如果出现，运行：
```bash
ls -lh experiments/models/*/dmd-*.pth | sort
# 应该看到4个不同的目录，每个目录一个.pth文件
```

---

## 下一步

完成smoke test后，进行完整训练：

```bash
# 监控进度
./run_all.sh --train 2>&1 | tee training.log

# 查看结果
tail -100 training.log  # 查看最后的结果汇总
```

最终会在 `experiments/models/` 下生成4个模型：
```
experiments/models/
├── mosi_aligned_bert/dmd-mosi.pth (Expected ACC7: 45.6%)
├── mosi_unaligned_bert/dmd-mosi.pth
├── mosei_aligned_bert/dmd-mosei.pth (Expected ACC7: 54.5%)
└── mosei_unaligned_bert/dmd-mosei.pth
```

---

## 重要说明

**🔴 注意：GloVe 实验不可复现**

数据文件的 'text' 字段存储的是 BERT (768维)，不是 GloVe (300维)。

如需复现 GloVe 结果，需要：
1. 从原始文本重新提取 GloVe 特征
2. 或从论文作者获取 GloVe 预处理的数据

当前可复现的论文结果：
- ✅ Table 1 - MOSI Aligned BERT (45.6%)
- ✅ Table 2 - MOSEI Aligned BERT (54.5%)
