# ✅ 问题已解决：数据文件不含 GloVe 特征

## 🔍 数据文件检查结果

已通过 `scripts/check_data.py` 检查数据文件，结果：

### 数据文件内容
```
text: (1284, 50, 768)        ← 名为 'text' 但存的是768维BERT特征！
text_bert: (1284, 3, 50)     ← 另一种BERT格式（可能是token级别）
```

### 🔴 **根本原因**
**数据文件的 'text' key 存储的是 BERT 特征，不是 GloVe！**

- 所有 .pkl 文件的 'text' 字段都是 768维 BERT 特征
- 没有 300维 GloVe 特征
- 原始 config.json 全部使用 768维 + use_bert=true，证实了这点

## 📊 影响

### 可复现的实验（4个）
✅ 只能复现 BERT 实验（论文 Table 1 & 2 中带 * 的行）：
1. mosi_aligned_bert - Expected ACC7: 45.6%
2. mosi_unaligned_bert
3. mosei_aligned_bert - Expected ACC7: 54.5%
4. mosei_unaligned_bert

### 不可复现的实验（原计划的2个）
❌ GloVe 实验无法复现（Table 1 & 2 中不带 * 的行）：
- mosi_aligned_glove - 论文报告 ACC7: 41.4%
- mosei_aligned_glove - 论文报告 ACC7: 53.7%

**原因：** 公开数据集不包含 GloVe 预处理特征。论文作者可能使用了未公开的 GloVe 数据处理流程。

## ✅ 解决方案

### 已修改的文件

1. **scripts/config_generator.py**
   - 移除了所有 GloVe 实验配置
   - 只生成 4 个 BERT 实验配置
   - 添加了说明注释

2. **后续需要更新**
   - scripts/batch_train.py（自动适应新的实验数量）
   - scripts/smoke_test.py（自动适应新的实验数量）
   - experiments/README.md（更新文档说明）

## 🚀 下一步

运行更新后的配置生成器：
```bash
cd DMD
python3 scripts/config_generator.py
```

应该生成 4 个配置文件，全部使用 BERT (768-dim)。

## 📝 技术细节

### 为什么会产生维度错误

**当 use_bert=False 时：**
```python
# data_loader.py 期望加载 GloVe
self.text = data[self.mode]['text']  # 期望 (N, 50, 300)
# 但实际得到 (N, 50, 768) - BERT！

# model 初始化为 300 维卷积
Conv1d(300, ..., kernel_size=5)

# 输入是 768 维
# 错误：expected 300 channels, got 768
```

### 数据文件的名称误导性

```python
data = {
    'text': np.array(..., shape=(N, 50, 768)),      # 误导：名为text但是BERT
    'text_bert': np.array(..., shape=(N, 3, 50)),   # 另一种BERT格式
    'vision': ...,
    'audio': ...
}
```

'text' 这个名字让人以为是 GloVe，但实际存储的是 BERT 特征。
