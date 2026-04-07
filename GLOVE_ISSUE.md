# 关键问题分析：GloVe 特征不可用

## 🔴 问题根源

### 错误信息
```
Given groups=1, weight of size [50, 300, 5], 
expected input[16, 768, 50] to have 300 channels, but got 768 channels instead
```

**翻译：**
- 模型期待 300 维输入（GloVe）
- 实际得到 768 维输入（BERT）
- 配置说 use_bert=False，但数据加载器仍加载了 BERT 特征

## 🔍 源码分析（已完成）

### 1. 配置加载逻辑（config.py）
```python
# config.py 第 27 行
dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] 
                and 'aligned' in dataset_args) else dataset_args['unaligned']

# 第 32-34 行 - 合并顺序
config.update(dataset_args)        # 包含 feature_dims
config.update(model_common_args)   # 包含 use_bert
config.update(model_dataset_args)  # 模型特定参数
```

✅ 配置加载逻辑正确，use_bert=False 确实被设置

### 2. 数据加载逻辑（data_loader.py）
```python
# data_loader.py 第 22-25 行
if 'use_bert' in self.args and self.args['use_bert']:
    self.text = data[self.mode]['text_bert'].astype(np.float32)  # BERT
else:
    self.text = data[self.mode]['text'].astype(np.float32)  # GloVe
```

✅ 数据加载逻辑正确，use_bert=False 时应该加载 'text'

### 3. 原始配置文件（config/config.json）
```json
{
  "datasetCommonParams": {
    "mosi": {
      "aligned": {
        "feature_dims": [768, 5, 20]  // ← 问题在这里！全是 768
      },
      "unaligned": {
        "feature_dims": [768, 5, 20]  // ← 都是 768
      }
    }
  },
  "dmd": {
    "commonParams": {
      "use_bert": true  // ← 原始默认是 true
    }
  }
}
```

❌ **原始配置全是 768，说明原仓库默认只支持 BERT**

## 💡 结论

### 数据文件可能的情况

**情况 1：数据文件只包含 BERT 特征**
```python
data['train'].keys() = ['text_bert', 'vision', 'audio', ...]
# 没有 'text' (GloVe)
```

如果是这种情况：
- ❌ **无法运行 GloVe 实验**
- ✅ 只能运行 BERT 实验（2个，不是6个）
- 论文报告的 GloVe 结果可能使用了不同的数据预处理

**情况 2：数据文件同时包含两种特征**
```python
data['train'].keys() = ['text', 'text_bert', 'vision', 'audio', ...]
# text.shape = (N, 50, 300)  - GloVe
# text_bert.shape = (N, 50, 768)  - BERT
```

如果是这种情况：
- ✅ 可以运行所有 6 个实验
- 当前错误是因为原始 config.json 的 feature_dims 设置错误

## 🚀 下一步行动

### 立即执行：检查数据文件

**在服务器上运行：**
```bash
cd /path/to/DMD
python3 scripts/check_data.py
```

这会检查数据文件中是否同时包含 'text' 和 'text_bert'。

### 根据检查结果的处理方案

#### 方案 A：如果数据只有 BERT（text_bert）

**修改实验计划：**
- 只运行 2 个 BERT 实验，不运行 GloVe
  - mosi_aligned_bert
  - mosei_aligned_bert
- 需要修改 scripts/ 中的实验列表
- 更新文档说明只能复现 BERT 结果

#### 方案 B：如果数据同时有 GloVe 和 BERT

**修改原始配置：**
```bash
# 需要根据 use_bert 动态设置 feature_dims
# 但这需要修改 config.py 或在 run.py 中动态调整
```

**临时解决方案：**
修改 config_generator.py，不在 datasetCommonParams 中设置 feature_dims，
让 data_loader.py 自动根据实际数据形状设置（第 39 行的逻辑）。

## 📋 需要用户确认

请在服务器上运行：
```bash
python3 scripts/check_data.py
```

然后告诉我结果，我会根据实际情况提供精确的修复方案。

## 🎯 可能的最终结论

根据原始 config.json 全是 768 和 use_bert=true 的事实，
**原仓库很可能只提供了 BERT 预处理的数据**。

论文中的 GloVe 结果可能：
1. 使用了未公开的 GloVe 预处理数据
2. 或者需要自己从原始数据重新提取 GloVe 特征

如果确实如此，我们只能复现论文 Table 1 和 Table 2 中带 * 的 BERT 实验。
