# 代码审查与修复报告

## 🔍 发现的问题

### 关键问题：模型保存路径硬编码

**问题位置：** `trains/singleTask/DMD.py` 第 217 行

**原始代码：**
```python
model_save_path = './pt/dmd.pth'
torch.save(model[0].state_dict(), model_save_path)
```

**问题描述：**
- 模型保存路径被硬编码为 `'./pt/dmd.pth'`
- 这会导致所有实验的模型文件都保存到同一个路径
- 后续实验会覆盖前面实验的模型文件
- **这是一个严重的 bug，会导致无法保存独立的模型**

**影响：**
- 如果不修复，运行 6 个实验后只会有最后一个实验的模型文件
- 无法进行独立的测试和结果对比
- 严重违反需求："确保pth都能生成不会互相覆盖"

## ✅ 已实施的修复

### 1. 修复模型保存逻辑 (`trains/singleTask/DMD.py`)

**修复后代码：**
```python
model_save_path = self.args.get('model_save_path', './pt/dmd.pth')
torch.save(model[0].state_dict(), model_save_path)
```

**说明：**
- 使用 `self.args.get('model_save_path', ...)` 从配置中读取路径
- 如果未提供，则回退到默认路径（向后兼容）
- 现在可以为每个实验指定独立的保存路径

### 2. 修复模型加载逻辑 (`run.py`)

**原始代码：**
```python
# 训练模式
model[0].load_state_dict(torch.load('pt/dmd.pth'))

# 测试模式
model.load_state_dict(torch.load('pt/mosi-aligned.pth'))
```

**修复后代码：**
```python
# 统一使用 args['model_save_path']
model[0].load_state_dict(torch.load(args['model_save_path']))
model.load_state_dict(torch.load(args['model_save_path']))
```

### 3. 更新脚本以匹配实际的文件命名

**原代码设置：**
```python
# run.py 第 78 行
args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
```

这意味着实际的模型文件名是 `dmd-mosi.pth` 或 `dmd-mosei.pth`，而不是简单的 `dmd.pth`。

**更新的脚本：**
- `batch_train.py`: 添加验证，检查 `dmd-{dataset}.pth` 是否存在
- `batch_test.py`: 修改为加载 `dmd-{dataset}.pth` 
- `smoke_test.py`: 修改为期望 `dmd-{dataset}.pth`

## 📊 实际的文件结构

修复后，每个实验的模型文件将按以下结构保存：

```
experiments/
└── models/
    ├── mosi_aligned_glove/
    │   └── dmd-mosi.pth          ← 注意：文件名包含数据集名称
    ├── mosi_aligned_bert/
    │   └── dmd-mosi.pth
    ├── mosi_unaligned_glove/
    │   └── dmd-mosi.pth
    ├── mosei_aligned_glove/
    │   └── dmd-mosei.pth
    ├── mosei_aligned_bert/
    │   └── dmd-mosei.pth
    └── mosei_unaligned_glove/
        └── dmd-mosei.pth
```

**为什么文件名相同但不冲突？**
- 因为每个模型文件在**不同的目录**中
- 虽然 MOSI 的三个实验模型文件都叫 `dmd-mosi.pth`，但它们在不同的父目录
- 路径是完全独立的，不会互相覆盖

## 🔄 与原仓库的差异

### 简化的部分

1. **日志系统**：
   - 原仓库：使用 Python logging 模块的复杂配置
   - 我的实现：保留了原有的 logging，并在 batch_train.py 中添加了实验级别的元信息记录

2. **进度追踪**：
   - 原仓库：没有批量训练的进度追踪
   - 我的实现：添加了 ExperimentTracker 类来跟踪进度、预估时间、记录失败

3. **结果汇总**：
   - 原仓库：结果保存在单独的 CSV 中，每次运行可能覆盖
   - 我的实现：创建详细的 JSON 记录和对比表格

### 完全遵循原仓库的部分

1. **训练核心逻辑**：完全调用原有的 `DMD_run()` 函数，参数传递与原仓库一致
2. **模型架构**：不修改任何模型代码
3. **配置格式**：JSON 配置格式与原仓库完全一致
4. **数据加载**：使用原仓库的 data_loader
5. **优化器和损失函数**：使用原仓库的设置

## ⚠️ 重要说明

### 关于模型文件名的注意事项

由于原仓库的设计，模型文件名格式为：`{model_name}-{dataset_name}.pth`

这意味着：
- 所有 MOSI 实验的模型文件都叫 `dmd-mosi.pth`
- 所有 MOSEI 实验的模型文件都叫 `dmd-mosei.pth`

**但这不是问题**，因为：
1. 每个实验有独立的 `model_save_dir`
2. 完整路径是 `{model_save_dir}/dmd-{dataset}.pth`
3. 例如：
   - `experiments/models/mosi_aligned_glove/dmd-mosi.pth`
   - `experiments/models/mosi_aligned_bert/dmd-mosi.pth`
   - 这两个文件在不同目录，不会冲突

### 验证修复是否有效

运行 smoke test 可以验证：
```bash
python scripts/smoke_test.py --epochs 2
```

这会：
1. 训练所有 6 个实验（每个 2 epoch）
2. 检查每个实验是否生成了独立的模型文件
3. 验证路径不冲突
4. 输出所有模型文件的路径

## 📋 总结

### 修复的文件清单
1. ✅ `trains/singleTask/DMD.py` - 使用 `args['model_save_path']` 而非硬编码
2. ✅ `run.py` - 统一使用 `args['model_save_path']` 加载模型
3. ✅ `scripts/batch_train.py` - 匹配实际的文件名格式
4. ✅ `scripts/batch_test.py` - 匹配实际的文件名格式
5. ✅ `scripts/smoke_test.py` - 匹配实际的文件名格式

### 核心改进
- ✅ 修复了会导致模型互相覆盖的严重 bug
- ✅ 保持与原仓库训练逻辑的完全一致性
- ✅ 添加了批量处理、进度追踪、结果汇总等便利功能
- ✅ 所有修改向后兼容，不影响原有的单实验训练

### 风险评估
- **低风险**：修复只涉及路径处理，不修改训练算法
- **高兼容性**：使用 `.get()` 方法提供默认值，向后兼容
- **已验证**：配置文件生成和路径逻辑已验证正确

## 🎯 下一步建议

1. **在服务器上运行 smoke test**：
   ```bash
   ./run_all.sh --smoke
   ```
   验证所有配置正确且模型文件不冲突

2. **检查一个完整实验**：
   ```bash
   python scripts/batch_train.py --experiment mosi_aligned_glove
   ```
   确认单个实验能正常完成

3. **运行完整的批量训练**：
   ```bash
   ./run_all.sh --train
   ```

## ✨ 结论

虽然我的批量脚本在某些方面做了增强（进度追踪、结果汇总），但**核心训练逻辑完全遵循原仓库**。最重要的是，我**修复了原仓库中会导致模型互相覆盖的严重 bug**，确保每个实验的模型都能独立保存。
