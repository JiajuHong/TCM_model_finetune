"""任务2工具函数：评估度量、处方后处理和约束"""

import re
import numpy as np
from collections import Counter

def normalize_herbs(herbs_str):
    """规范化中药字符串为列表"""
    try:
        # 尝试解析可能是字符串形式的列表
        if isinstance(herbs_str, str):
            # 处理可能是字符串形式的列表
            if herbs_str.startswith('[') and herbs_str.endswith(']'):
                try:
                    herbs = eval(herbs_str)
                    if isinstance(herbs, list):
                        return [h.strip() for h in herbs]
                except:
                    pass
            
            # 处理可能是逗号分隔的列表
            herbs = [h.strip() for h in re.split(r'[,，、]', herbs_str) if h.strip()]
            return herbs
        elif isinstance(herbs_str, list):
            return [h.strip() for h in herbs_str]
        else:
            return []
    except:
        return []

def filter_herbs(herbs, valid_herbs_set):
    """过滤不在有效中药列表中的药物"""
    return [herb for herb in herbs if herb in valid_herbs_set]

def postprocess_herbs(herbs_str, valid_herbs_set, min_herbs=10, max_herbs=15):
    """处理和约束中药列表"""
    # 规范化为列表
    herbs = normalize_herbs(herbs_str)
    
    # 过滤不在有效列表中的药物
    valid_herbs = filter_herbs(herbs, valid_herbs_set)
    
    # 去重
    valid_herbs = list(dict.fromkeys(valid_herbs))
    
    # 确保药物数量在指定范围内
    if len(valid_herbs) < min_herbs:
        # 如果药物数量不足，可以不处理，交给评估函数计算分数
        pass
    elif len(valid_herbs) > max_herbs:
        # 如果药物数量过多，截取前max_herbs个
        valid_herbs = valid_herbs[:max_herbs]
    
    return valid_herbs

def calculate_jaccard(true_herbs, pred_herbs):
    """计算Jaccard相似系数"""
    true_set = set(true_herbs)
    pred_set = set(pred_herbs)
    
    intersection = len(true_set.intersection(pred_set))
    union = len(true_set.union(pred_set))
    
    if union == 0:
        return 0
    
    return intersection / union

def calculate_recall(true_herbs, pred_herbs):
    """计算召回率"""
    true_set = set(true_herbs)
    pred_set = set(pred_herbs)
    
    intersection = len(true_set.intersection(pred_set))
    
    if len(true_set) == 0:
        return 0
    
    return intersection / len(true_set)

def calculate_precision(true_herbs, pred_herbs):
    """计算精确率"""
    true_set = set(true_herbs)
    pred_set = set(pred_herbs)
    
    intersection = len(true_set.intersection(pred_set))
    
    if len(pred_set) == 0:
        return 0
    
    return intersection / len(pred_set)

def calculate_f1(true_herbs, pred_herbs):
    """计算F1分数"""
    precision = calculate_precision(true_herbs, pred_herbs)
    recall = calculate_recall(true_herbs, pred_herbs)
    
    if precision + recall == 0:
        return 0
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_avg_herbs(true_herbs, pred_herbs):
    """计算药物平均数量匹配度"""
    true_count = len(true_herbs)
    pred_count = len(pred_herbs)
    
    if max(true_count, pred_count) == 0:
        return 0
    
    return 1 - abs(true_count - pred_count) / max(true_count, pred_count)

def evaluate_prescription(true_herbs, pred_herbs):
    """综合评估中药处方"""
    # 计算各项指标
    jaccard = calculate_jaccard(true_herbs, pred_herbs)
    recall = calculate_recall(true_herbs, pred_herbs)
    precision = calculate_precision(true_herbs, pred_herbs)
    f1 = calculate_f1(true_herbs, pred_herbs)
    avg_herbs = calculate_avg_herbs(true_herbs, pred_herbs)
    
    # 计算任务2总分
    task2_score = (jaccard + f1 + avg_herbs) / 3
    
    return {
        "jaccard": jaccard,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "avg_herbs": avg_herbs,
        "task2_score": task2_score
    }

def select_best_candidate(candidates, reference, valid_herbs_set, min_herbs=10, max_herbs=15):
    """从多个候选处方中选择最好的一个"""
    best_score = -1
    best_candidate = None
    
    for candidate in candidates:
        # 处理候选处方
        processed_herbs = postprocess_herbs(candidate, valid_herbs_set, min_herbs, max_herbs)
        
        # 如果没有有效药物，跳过
        if not processed_herbs:
            continue
        
        # 评估当前候选处方
        if reference:
            ref_herbs = normalize_herbs(reference)
            metrics = evaluate_prescription(ref_herbs, processed_herbs)
            score = metrics["task2_score"]
        else:
            # 如果没有参考处方，使用药物数量作为评分标准
            herbs_count = len(processed_herbs)
            # 分数是处方中药数量接近理想范围中点的程度
            ideal_count = (min_herbs + max_herbs) / 2
            score = 1 - abs(herbs_count - ideal_count) / ideal_count
        
        # 更新最佳候选处方
        if score > best_score:
            best_score = score
            best_candidate = processed_herbs
    
    return best_candidate if best_candidate else []

def prepare_submission(predictions, output_file):
    """准备提交格式的预测结果"""
    submission = []
    
    for id, herbs in predictions.items():
        submission.append({
            "ID": id,
            "子任务1": [],
            "子任务2": herbs
        })
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
    
    print(f"Saved submission to {output_file}")
    return submission 