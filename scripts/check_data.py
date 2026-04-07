#!/usr/bin/env python3
"""
数据文件检查工具 - 在服务器上运行此脚本检查数据文件内容
"""
import pickle
import sys

def check_data_file(filepath):
    """检查数据文件中的特征"""
    print(f"检查文件: {filepath}")
    print("=" * 80)
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print("\n可用的split:")
        print(list(data.keys()))
        
        print("\ntrain split中的key:")
        train_keys = list(data['train'].keys())
        print(train_keys)
        
        print("\n特征维度:")
        for key in train_keys:
            if hasattr(data['train'][key], 'shape'):
                print(f"  {key}: {data['train'][key].shape}")
        
        # 检查是否同时有text和text_bert
        has_glove = 'text' in train_keys
        has_bert = 'text_bert' in train_keys
        
        print("\n特征可用性:")
        print(f"  GloVe (text): {'✓' if has_glove else '✗'}")
        print(f"  BERT (text_bert): {'✓' if has_bert else '✗'}")
        
        if has_glove:
            print(f"\n  GloVe特征维度: {data['train']['text'].shape}")
        if has_bert:
            print(f"  BERT特征维度: {data['train']['text_bert'].shape}")
            
        return has_glove, has_bert
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False, False

if __name__ == "__main__":
    print("数据文件内容检查工具")
    print("=" * 80)
    
    files_to_check = [
        "dataset/MOSI/Processed/aligned_50.pkl",
        "dataset/MOSI/Processed/unaligned_50.pkl",
        "dataset/MOSEI/Processed/aligned_50.pkl",
        "dataset/MOSEI/Processed/unaligned_50.pkl",
    ]
    
    results = {}
    for filepath in files_to_check:
        print(f"\n{'='*80}")
        has_glove, has_bert = check_data_file(filepath)
        results[filepath] = {'glove': has_glove, 'bert': has_bert}
        print()
    
    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    for filepath, result in results.items():
        glove_status = "✓" if result['glove'] else "✗"
        bert_status = "✓" if result['bert'] else "✗"
        print(f"{filepath}")
        print(f"  GloVe: {glove_status}  BERT: {bert_status}")
