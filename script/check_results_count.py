#!/usr/bin/env python3
"""
检测JSON文件中experiments的results个数

用法：
  python check_results_count.py <json_file_path>
  python check_results_count.py <directory_path>  # 检测目录下所有json文件
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def check_file(file_path: Path) -> dict:
    """检测单个JSON文件中的results个数"""
    try:
        data = json.loads(file_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        return {'error': f'JSON解析失败: {e}'}
    except Exception as e:
        return {'error': f'读取文件失败: {e}'}
    
    if not isinstance(data, dict):
        return {'error': 'JSON根元素必须是字典'}
    
    experiments = data.get('experiments', [])
    if not isinstance(experiments, list):
        return {'error': 'experiments字段必须是列表'}
    
    results = {
        'industry': data.get('industry', 'Unknown'),
        'total_experiments': len(experiments),
        'experiments': []
    }
    
    total_results = 0
    for idx, exp in enumerate(experiments, 1):
        if not isinstance(exp, dict):
            results['experiments'].append({
                'index': idx,
                'name': 'N/A',
                'results_count': 0,
                'error': '实验项必须是字典'
            })
            continue
        
        exp_name = exp.get('name', f'exp_{idx}')
        exp_results = exp.get('results', [])
        
        if not isinstance(exp_results, list):
            results['experiments'].append({
                'index': idx,
                'name': exp_name,
                'results_count': 0,
                'error': 'results字段必须是列表'
            })
            continue
        
        results_count = len(exp_results)
        total_results += results_count
        
        results['experiments'].append({
            'index': idx,
            'name': exp_name,
            'results_count': results_count
        })
    
    results['total_results'] = total_results
    return results


def print_results(file_path: Path, results: dict):
    """打印检测结果"""
    print(f"\n📄 文件: {file_path.name}")
    print("=" * 60)
    
    if 'error' in results:
        print(f"❌ 错误: {results['error']}")
        return
    
    print(f"行业: {results['industry']}")
    print(f"Experiments总数: {results['total_experiments']}")
    print(f"Results总数: {results['total_results']}\n")
    
    print("详细信息:")
    print(f"{'#':<4} {'Experiment名称':<40} {'Results个数':>10}")
    print("-" * 60)
    
    for exp in results['experiments']:
        if 'error' in exp:
            print(f"{exp['index']:<4} {exp['name']:<40} ❌ {exp['error']}")
        else:
            print(f"{exp['index']:<4} {exp['name']:<40} {exp['results_count']:>10}")


def main():
    if len(sys.argv) < 2:
        print("用法: python check_results_count.py <json文件或目录路径>")
        print("\n示例:")
        print("  python check_results_count.py cv_scores_exp1_run1.json")
        print("  python check_results_count.py ./result/qwen/")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if not path.exists():
        print(f"❌ 路径不存在: {path}")
        sys.exit(1)
    
    files_to_check = []
    
    if path.is_file():
        if path.suffix.lower() == '.json':
            files_to_check = [path]
        else:
            print(f"❌ 文件不是JSON格式: {path}")
            sys.exit(1)
    elif path.is_dir():
        files_to_check = sorted(path.glob('*.json'))
        if not files_to_check:
            print(f"❌ 目录中没有JSON文件: {path}")
            sys.exit(1)
    
    print(f"\n🔍 检测 {len(files_to_check)} 个文件...\n")
    
    # 统计汇总
    summary = {
        'total_files': len(files_to_check),
        'valid_files': 0,
        'error_files': 0,
        'total_results_all': 0
    }
    
    for file_path in files_to_check:
        results = check_file(file_path)
        print_results(file_path, results)
        
        if 'error' in results:
            summary['error_files'] += 1
        else:
            summary['valid_files'] += 1
            summary['total_results_all'] += results['total_results']
    
    # 打印总结
    print("\n" + "=" * 60)
    print("📊 汇总统计")
    print("=" * 60)
    print(f"检测文件总数: {summary['total_files']}")
    print(f"有效文件: {summary['valid_files']}")
    print(f"错误文件: {summary['error_files']}")
    print(f"全部Results总数: {summary['total_results_all']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
