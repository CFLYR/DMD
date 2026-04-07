# DMD Test Results Fix

## 问题描述

运行 `./run_all.sh --test` 时，测试成功执行并打印了结果，但 batch_test.py 判定所有实验失败：

```
Failed Experiments:
  ✗ mosi_aligned_bert: None
  ✗ mosei_aligned_bert: None
```

## 根本原因

有两个关键问题：

### 1. DMD_run 函数不返回结果

**位置：** `run.py` 第 88-113 行

**问题：** DMD_run 函数内部收集了 `model_results` 列表，但函数结束时没有 `return` 语句，导致返回 `None`。

**修复：** 在函数末尾添加：
```python
# Return results for programmatic access
# For single seed, return the result directly; for multiple seeds, return first result
return model_results[0] if len(model_results) == 1 else model_results[0]
```

### 2. 测试模式有交互式阻塞

**位置：** `run.py` 第 173-177 行的 `_run` 函数

**问题：** 在测试模式下有 `input('[Press Any Key to start another run]')`，这会阻塞自动化脚本。

**修复：** 添加条件判断，只在详细模式下才等待输入：
```python
if args.mode == 'test':
    model.load_state_dict(torch.load(args['model_save_path']))
    results = trainer.do_test(model, dataloader['test'], mode="TEST")
    sys.stdout.flush()
    # Only wait for input if in interactive mode (verbose_level >= 2)
    if args.get('verbose_level', 1) >= 2:
        input('[Press Any Key to start another run]')
```

### 3. batch_test.py 的结果解析不正确

**位置：** `scripts/batch_test.py` 第 158-169 行

**问题：** 使用了错误的指标键名尝试匹配，实际的键名是：
- `Acc_7` (7-class accuracy, 0-1 范围)
- `Acc_2` (2-class accuracy, 0-1 范围)
- `F1_score` (F1 score, 0-1 范围)
- `MAE` (Mean Absolute Error)
- `Loss`

**修复：** 使用正确的键名映射并转换为百分比：
```python
metric_mapping = {
    "ACC7": "Acc_7",
    "ACC2": "Acc_2", 
    "F1": "F1_score"
}

for paper_metric, result_key in metric_mapping.items():
    result_value = None
    if result_key in result:
        # Convert from 0-1 range to percentage
        result_value = result[result_key] * 100
```

## 数据流

修复后的正确数据流：

1. **batch_test.py** 调用 `DMD_run()` 
2. **DMD_run()** 循环调用 `_run()` 收集 `model_results`
3. **_run()** 调用 `trainer.do_test()` 并返回 `results` 字典
4. **do_test()** 调用 `self.metrics()` 返回 `eval_results` 字典
5. **DMD_run()** 返回 `model_results[0]` 给调用者
6. **batch_test.py** 解析返回的字典并比较论文结果

## 指标键名参考

从 `trains/utils/metricsTop.py` 的 `__eval_mosei_regression()` 函数：

```python
eval_results = {
    "Acc_2":  round(non_zeros_acc2, 4),      # 0-1 范围
    "F1_score": round(non_zeros_f1_score, 4), # 0-1 范围
    "Acc_7": round(mult_a7, 4),               # 0-1 范围
    "MAE": round(mae, 4),                     # 原始值
}
```

## 测试方法

### 方法 1：快速验证（本地）

```bash
cd /Users/mac/Documents/2025tongji/Traoz/UES/DMD
python test_fix.py
```

应该看到：
```
✓ SUCCESS! DMD_run now returns results properly

Keys in result:
  Acc_2: 0.8369
  F1_score: 0.8366
  Acc_7: 0.4504
  MAE: 0.7389
  Loss: 0.7383
```

### 方法 2：完整批量测试（服务器）

```bash
./run_all.sh --test
```

应该看到：
```
TESTING SUMMARY
================================================================================
Total: 2
Successful: 2
Failed: 0
```

## 修改的文件

1. `run.py`
   - 第 114 行添加：返回 `model_results[0]`
   - 第 177 行修改：条件性 input() 阻塞

2. `scripts/batch_test.py`
   - 第 120 行修改：`if result is not None:` 而不是 `if result:`
   - 第 158-169 行重写：使用正确的键名映射

3. `test_fix.py`（新增）
   - 用于快速验证修复是否生效

## 注意事项

- **禁止使用输出捕获**：修复方案直接使用函数返回值，而不是解析 stdout
- **自动化友好**：verbose_level=1 时不会有交互式阻塞
- **单种子返回**：使用单个种子 [1111] 时，返回该种子的结果字典
- **类型安全**：batch_test 现在检查 `result is not None` 而不是 `if result`（防止空字典被误判为 False）
