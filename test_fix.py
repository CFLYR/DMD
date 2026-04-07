#!/usr/bin/env python3
"""
Quick test to verify DMD_run returns results properly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run import DMD_run

# Test with MOSI
print("Testing DMD_run return value...")
print("=" * 80)

result = DMD_run(
    model_name='dmd',
    dataset_name='mosi',
    config_file='experiments/configs/mosi_aligned_bert.json',
    seeds=[1111],
    model_save_dir='experiments/models/mosi_aligned_bert',
    res_save_dir='experiments/results',
    log_dir='experiments/logs',
    mode='test',
    is_distill=False,
    verbose_level=1  # Non-interactive mode
)

print("\n" + "=" * 80)
print("Return value type:", type(result))
print("Return value:", result)

if result is not None and isinstance(result, dict):
    print("\n✓ SUCCESS! DMD_run now returns results properly")
    print("\nKeys in result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
else:
    print("\n✗ FAILED! DMD_run did not return a dict")
    sys.exit(1)
