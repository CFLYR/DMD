# 数据集特征分析报告

运行 `python3 scripts/check_data.py` 的完整分析结果

## 📊 数据文件结构

### MOSI/Processed/aligned_50.pkl
```
Splits: ['train', 'valid', 'test']
Keys: ['raw_text', 'audio', 'vision', 'id', 'text', 'text_bert', 
       'annotations', 'classification_labels', 'regression_labels']

特征维度（train split）:
  raw_text:              (1284,)        - 原始文本
  audio:                 (1284, 50, 5)  - 对齐到50帧
  vision:                (1284, 50, 20) - 对齐到50帧
  text:                  (1284, 50, 768) ⚠️  名为text但是BERT！
  text_bert:             (1284, 3, 50)   - Token级BERT
  classification_labels: (1284,)
  regression_labels:     (1284,)
```

### MOSI/Processed/unaligned_50.pkl
```
Splits: ['train', 'valid', 'test']
Keys: ['raw_text', 'audio', 'vision', 'id', 'text', 'text_bert',
       'audio_lengths', 'vision_lengths', 'annotations', 
       'classification_labels', 'regression_labels']

特征维度（train split）:
  raw_text:              (1284,)
  audio:                 (1284, 375, 5)   - 变长（最大375）
  vision:                (1284, 500, 20)  - 变长（最大500）
  text:                  (1284, 50, 768) ⚠️  名为text但是BERT！
  text_bert:             (1284, 3, 50)
  audio_lengths:         (1284,)          - 实际音频长度
  vision_lengths:        (1284,)          - 实际视频长度
  classification_labels: (1284,)
  regression_labels:     (1284,)
```

### MOSEI/Processed/aligned_50.pkl
```
Splits: ['train', 'valid', 'test']
Keys: ['raw_text', 'audio', 'vision', 'id', 'text', 'text_bert',
       'annotations', 'classification_labels', 'regression_labels']

特征维度（train split）:
  raw_text:              (16326,)
  audio:                 (16326, 50, 74)
  vision:                (16326, 50, 35)
  text:                  (16326, 50, 768) ⚠️  名为text但是BERT！
  text_bert:             (16326, 3, 50)
  classification_labels: (16326,)
  regression_labels:     (16326,)
```

### MOSEI/Processed/unaligned_50.pkl
```
Splits: ['train', 'valid', 'test']
Keys: ['raw_text', 'audio', 'vision', 'id', 'text', 'text_bert',
       'audio_lengths', 'vision_lengths', 'annotations',
       'classification_labels', 'regression_labels']

特征维度（train split）:
  raw_text:              (16326,)
  audio:                 (16326, 500, 74)  - 变长（最大500）
  vision:                (16326, 500, 35)  - 变长（最大500）
  text:                  (16326, 50, 768) ⚠️  名为text但是BERT！
  text_bert:             (16326, 3, 50)
  audio_lengths:         (16326,)
  vision_lengths:        (16326,)
  classification_labels: (16326,)
  regression_labels:     (16326,)
```

## 🔍 关键发现

### 1. **'text' 字段存储的是 BERT 特征，不是 GloVe**

所有数据文件中：
- `'text'` shape = `(N, 50, 768)` - **768维是BERT的标准维度**
- `'text_bert'` shape = `(N, 3, 50)` - 可能是 [CLS, SEP, PAD] token的表示

**结论：数据集不包含 GloVe (300维) 特征！**

### 2. **对齐 vs 非对齐的区别**

**Aligned（对齐）：**
- 所有模态都填充/截断到固定长度（50帧）
- Audio: (N, **50**, feature_dim)
- Vision: (N, **50**, feature_dim)
- Text: (N, **50**, 768)

**Unaligned（非对齐）：**
- 保留原始变长序列
- Audio: (N, **max_len**, feature_dim) + audio_lengths
- Vision: (N, **max_len**, feature_dim) + vision_lengths
- Text: 仍然是 (N, 50, 768)，因为BERT输入本身已对齐

### 3. **数据集规模**

| Dataset | Train Samples | Feature Dims (A, V, T) |
|---------|---------------|------------------------|
| MOSI    | 1,284         | 5, 20, 768             |
| MOSEI   | 16,326        | 74, 35, 768            |

MOSEI 是 MOSI 的 ~12.7倍大小

## ❌ 不可复现的实验

由于数据集不包含 GloVe 特征，以下实验**无法复现**：

1. **MOSI Aligned GloVe** (Table 1 - DMD, ACC7: 41.4%)
2. **MOSI Unaligned GloVe** (Table 1 - DMD, ACC7: 41.9%)
3. **MOSEI Aligned GloVe** (Table 2 - DMD, ACC7: 53.7%)
4. **MOSEI Unaligned GloVe** (Table 2 - DMD, ACC7: 54.6%)

## ✅ 可复现的实验

只能复现 BERT 实验（论文中带 * 的行）：

1. **MOSI Aligned BERT** (Table 1 - DMD*, Expected ACC7: **45.6%**)
2. **MOSI Unaligned BERT** (未在 Table 1 报告)
3. **MOSEI Aligned BERT** (Table 2 - DMD*, Expected ACC7: **54.5%**)
4. **MOSEI Unaligned BERT** (未在 Table 2 报告)

## 💡 为什么论文有 GloVe 结果？

**可能的原因：**

1. **未公开的数据预处理**
   - 论文作者可能使用了不同的数据预处理流程
   - GloVe 特征需要从原始文本重新提取

2. **不同的数据源**
   - 可能使用了 CMU-Multimodal SDK 的原始数据
   - 自己提取了 GloVe 特征（如使用预训练的 GloVe-840B）

3. **代码仓库的演化**
   - 当前公开的数据可能是后期更新的版本
   - 早期版本可能包含 GloVe 预处理

## 🔧 如果要复现 GloVe 实验

需要手动预处理：

1. 获取 CMU-MOSI 和 CMU-MOSEI 原始数据
2. 使用预训练 GloVe 模型（如 glove.840B.300d）
3. 对每个样本的文本进行：
   - 分词
   - GloVe embedding 查找
   - 对齐到固定长度（50）
4. 替换 .pkl 文件中的 'text' 字段

**工作量较大，建议只复现 BERT 实验。**

## 📝 代码修改说明

已修改 `scripts/config_generator.py`：
- 移除所有 GloVe 实验配置
- 只生成 4 个 BERT 实验配置
- 所有配置使用 `use_bert: true` 和 `feature_dims: [768, ...]`

下一步：
```bash
cd DMD
python3 scripts/config_generator.py  # 生成4个配置
./run_all.sh smoke                    # 快速验证（应该全部通过）
./run_all.sh train                    # 完整训练
```
