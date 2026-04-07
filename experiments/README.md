# DMD 实验复现指南

本目录包含复现 CVPR 2023 DMD 论文（Table 1 和 Table 2）实验结果的完整脚本系统。

## 📋 目录结构

```
experiments/
├── configs/              # 实验配置文件（自动生成）
├── models/               # 训练好的模型权重
│   ├── mosi_aligned_glove/
│   ├── mosi_aligned_bert/
│   ├── mosi_unaligned_glove/
│   ├── mosei_aligned_glove/
│   ├── mosei_aligned_bert/
│   └── mosei_unaligned_glove/
├── logs/                 # 训练日志
└── results/              # 实验结果
    ├── comparison_with_paper.csv
    └── training_results.json
```

## 🚀 快速开始

### 方式 1：一键运行（推荐）

```bash
# 1. 环境检查
./run_all.sh --check

# 2. 快速验证（2个epoch，约5分钟）
./run_all.sh --smoke

# 3. 完整训练（数小时）
./run_all.sh --train

# 4. 测试所有模型
./run_all.sh --test

# 5. 完整流程（验证+训练+测试）
./run_all.sh --all
```

### 方式 2：分步执行

```bash
# 1. 生成配置文件
python scripts/config_generator.py

# 2. 快速验证（可选，验证路径不冲突）
python scripts/smoke_test.py --epochs 2

# 3. 批量训练
python scripts/batch_train.py

# 4. 批量测试
python scripts/batch_test.py
```

## 📊 实验配置对照表

根据论文 Table 1 和 Table 2，共需运行 6 个实验：

| 实验ID | 数据集 | 对齐 | 语言特征 | 配置名称 | 预期ACC7 | 论文位置 |
|--------|--------|------|----------|----------|----------|----------|
| 1 | MOSI | ✓ | GloVe (300d) | `mosi_aligned_glove` | 41.4% | Table 1, Aligned |
| 2 | MOSI | ✓ | BERT* (768d) | `mosi_aligned_bert` | 45.6% | Table 1, Aligned* |
| 3 | MOSI | ✗ | GloVe (300d) | `mosi_unaligned_glove` | 41.9% | Table 1, Unaligned |
| 4 | MOSEI | ✓ | GloVe (300d) | `mosei_aligned_glove` | 53.7% | Table 2, Aligned |
| 5 | MOSEI | ✓ | BERT* (768d) | `mosei_aligned_bert` | 54.5% | Table 2, Aligned* |
| 6 | MOSEI | ✗ | GloVe (300d) | `mosei_unaligned_glove` | 54.6% | Table 2, Unaligned |

*注：带 * 表示使用 BERT 特征*

## 🔧 配置参数说明

每个实验的关键配置差异：

### GloVe vs BERT
- **GloVe**: `use_bert: false`, `feature_dims[0]: 300`
- **BERT**: `use_bert: true`, `feature_dims[0]: 768`

### Aligned vs Unaligned
- **Aligned**: `need_data_aligned: true`, 使用 `aligned_50.pkl`
- **Unaligned**: `need_data_aligned: false`, 使用 `unaligned_50.pkl`

### 固定参数
- 随机种子：`1111`
- Batch size：`16`
- Early stop：`10` epochs
- Learning rate：`0.0001`

## 📝 使用详解

### 1. 配置生成器 (config_generator.py)

自动生成 6 个实验的配置文件到 `experiments/configs/`。

```bash
python scripts/config_generator.py
```

**输出：**
- `experiments/configs/mosi_aligned_glove.json`
- `experiments/configs/mosi_aligned_bert.json`
- `experiments/configs/mosi_unaligned_glove.json`
- `experiments/configs/mosei_aligned_glove.json`
- `experiments/configs/mosei_aligned_bert.json`
- `experiments/configs/mosei_unaligned_glove.json`

### 2. 快速验证 (smoke_test.py)

在正式训练前，快速验证所有配置是否正确，模型保存路径是否冲突。

```bash
# 默认运行 2 个 epoch
python scripts/smoke_test.py

# 自定义 epoch 数
python scripts/smoke_test.py --epochs 3

# 保留验证文件（不清理）
python scripts/smoke_test.py --no-cleanup
```

**验证内容：**
- ✓ 每个实验都能成功启动
- ✓ 生成独立的 .pth 模型文件
- ✓ 路径不冲突，不会互相覆盖

### 3. 批量训练 (batch_train.py)

按顺序训练所有 6 个实验。每个实验的模型、日志、结果都保存到独立目录。

```bash
# 训练所有实验
python scripts/batch_train.py

# 从第 3 个实验继续（0-based index）
python scripts/batch_train.py --continue-from 2

# 只训练特定实验
python scripts/batch_train.py --experiment mosi_aligned_bert
```

**训练过程：**
- 显示详细进度（当前实验 X/6）
- 预估剩余时间
- 每个实验独立日志文件
- 失败的实验记录到 `failed_experiments.txt`
- 实验间等待 5 秒以释放 GPU 内存

**输出位置：**
- 模型：`experiments/models/{实验名}/dmd.pth`
- 日志：`experiments/logs/{实验名}.log`
- 结果：`experiments/results/training_results.json`

### 4. 批量测试 (batch_test.py)

测试所有训练好的模型，生成结果对比表。

```bash
# 测试所有模型
python scripts/batch_test.py

# 只测试特定模型
python scripts/batch_test.py --experiment mosi_aligned_bert
```

**输出：**
- 控制台显示对比表格
- CSV 文件：`experiments/results/comparison_with_paper.csv`
- JSON 详情：`experiments/results/test_results_detailed.json`

对比表格示例：
```
Experiment             Dataset  ACC7 (Our)  ACC7 (Paper)  ACC2 (Our)  ACC2 (Paper)
mosi_aligned_glove     MOSI     41.5 ✓      41.4          84.6 ✓      84.5
mosi_aligned_bert      MOSI     45.7 ✓      45.6          86.1 ✓      86.0
...
```

### 5. Shell 包装脚本 (run_all.sh)

一键式执行器，包含环境检查和完整流程。

```bash
# 检查环境（Python、GPU、依赖）
./run_all.sh --check

# 快速验证
./run_all.sh --smoke        # 默认 2 epochs
./run_all.sh --smoke 5      # 自定义 5 epochs

# 完整训练
./run_all.sh --train

# 测试模型
./run_all.sh --test

# 完整流程（验证→训练→测试）
./run_all.sh --all
```

**功能：**
- ✓ 自动检查 Python 版本
- ✓ 检测 GPU 可用性
- ✓ 验证必需的 Python 包
- ✓ 带颜色的状态输出
- ✓ 确认提示（训练前）

## 📈 结果解读

### 评价指标
- **ACC7**: 7 分类准确率（highly negative 到 highly positive）
- **ACC2**: 二分类准确率（negative vs non-negative）
- **F1**: F1 分数（二分类）

### 预期结果
训练完成后，各实验的 ACC7 应接近论文报告值（±0.5%）。

### 如果结果不匹配
1. 检查数据集文件是否正确加载
2. 确认配置参数（use_bert, need_data_aligned）
3. 查看训练日志中的 loss 曲线
4. 验证模型文件大小是否合理（~50-100MB）

## 🛠 故障排查

### 问题 1：配置文件生成失败
**症状：** `config_generator.py` 报错  
**解决：**
```bash
cd /path/to/DMD
python scripts/config_generator.py
```
确保当前在 DMD 根目录。

### 问题 2：模型路径冲突
**症状：** 训练后只有一个 `dmd.pth` 文件  
**解决：** 运行 smoke test 验证：
```bash
python scripts/smoke_test.py
```
检查输出是否报告 "All model paths are unique"。

### 问题 3：训练中断
**症状：** 训练到第 3 个实验时失败  
**解决：** 从失败的实验继续：
```bash
# 如果第 3 个实验（index=2）失败
python scripts/batch_train.py --continue-from 2
```

### 问题 4：GPU 内存不足
**症状：** CUDA out of memory  
**解决：**
1. 降低 batch_size（修改配置文件中的 `batch_size: 16` → `8`）
2. 确保实验间有足够延迟（batch_train.py 中已有 5 秒延迟）
3. 手动释放：
```python
import torch
torch.cuda.empty_cache()
```

### 问题 5：测试时找不到模型
**症状：** `batch_test.py` 报错 "Model not found"  
**解决：**
```bash
# 检查模型文件是否存在
ls -lh experiments/models/*/dmd.pth

# 确认训练完成
cat experiments/results/training_results.json
```

## 📚 高级用法

### 只训练特定实验
```bash
python scripts/batch_train.py --experiment mosei_aligned_bert
```

### 修改训练参数
编辑生成的配置文件：
```bash
vim experiments/configs/mosi_aligned_glove.json
# 修改 learning_rate、batch_size 等参数
```

### 查看详细日志
```bash
# 实时查看训练日志
tail -f experiments/logs/mosi_aligned_glove.log

# 查看所有日志
cat experiments/logs/*.log
```

### 比对多次运行结果
保留每次运行的结果目录：
```bash
cp -r experiments/results experiments/results_run1
# 运行新实验
cp -r experiments/results experiments/results_run2
```

## 🔍 文件说明

### 脚本文件
- `scripts/config_generator.py` - 配置生成器
- `scripts/batch_train.py` - 批量训练
- `scripts/batch_test.py` - 批量测试
- `scripts/smoke_test.py` - 快速验证
- `run_all.sh` - Shell 包装器

### 配置文件
- `experiments/configs/*.json` - 各实验配置

### 输出文件
- `experiments/models/*/dmd.pth` - 训练好的模型（每个约 50-100MB）
- `experiments/logs/*.log` - 训练日志
- `experiments/results/comparison_with_paper.csv` - 结果对比表
- `experiments/results/training_results.json` - 训练详情
- `experiments/results/test_results_detailed.json` - 测试详情
- `experiments/results/failed_experiments.txt` - 失败记录（如有）

## ⏱ 时间估计

| 阶段 | 预估时间 |
|------|---------|
| 配置生成 | < 1 分钟 |
| Smoke test | 5-10 分钟 |
| MOSI 训练 | 1-2 小时/实验 |
| MOSEI 训练 | 3-4 小时/实验 |
| 批量测试 | 10-20 分钟 |
| **总计** | **约 20-30 小时**（6个实验） |

*注：时间取决于 GPU 性能和数据集大小*

## 💡 提示

1. **先运行 smoke test**：避免长时间训练后发现配置错误
2. **监控 GPU 使用**：`watch -n 1 nvidia-smi`
3. **使用 tmux/screen**：长时间训练建议在后台会话中运行
4. **定期备份模型**：训练完成后备份 `experiments/models/`
5. **查看日志**：出错时先检查 `experiments/logs/` 中的日志

## 📧 问题反馈

如遇到问题，请提供：
1. 运行的命令
2. 错误信息（完整的 traceback）
3. 相关日志文件（`experiments/logs/`）
4. 环境信息（`python --version`, `nvidia-smi`）

---

**祝实验顺利！🎉**
