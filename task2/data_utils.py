"""数据处理模块：处理训练数据和验证数据，实现数据增强"""

import json
import re
import random
import os
import torch
import logging
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np

def load_herbs_list(herbs_file):
    """加载中药列表"""
    try:
        with open(herbs_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 尝试判断文件格式
            if content.startswith('[') and content.endswith(']'):
                # 如果是以[开头]结尾的格式，尝试解析为Python列表
                try:
                    herbs = eval(content)
                    # 清理药物名称（去除空格等）
                    herbs = [herb.strip() for herb in herbs if herb.strip()]
                    return herbs
                except Exception as e:
                    logging.warning(f"Failed to parse herbs list as Python list: {e}")
            
            # 标准格式：每行一个中药名
            lines = content.split('\n')
            herbs = [line.strip() for line in lines if line.strip()]
            
            if not herbs:
                # 如果上面的方法都没有得到药物列表，尝试按逗号分隔
                herbs = [item.strip() for item in content.split(',') if item.strip()]
        
        logging.info(f"Loaded {len(herbs)} herbs from {herbs_file}")
        return herbs
    except Exception as e:
        logging.error(f"Error loading herbs list: {e}")
        # 返回默认的空列表
        return []

def normalize_herb_name(herb):
    """规范化中药名称，去除空格和特殊字符"""
    herb = herb.strip()
    herb = re.sub(r'\s+', '', herb)
    return herb

def load_dataset(file_path, herbs_list=None):
    """加载数据集"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 规范化中药名称
        if herbs_list is not None:
            herbs_set = set(herbs_list)
            for item in data:
                if '处方' in item:
                    # 处理处方列表
                    if isinstance(item['处方'], str):
                        try:
                            herbs = eval(item['处方'])
                        except:
                            herbs = [item['处方']]
                    else:
                        herbs = item['处方']
                    
                    # 规范化并过滤处方
                    normalized_herbs = []
                    for herb in herbs:
                        norm_herb = normalize_herb_name(herb)
                        if norm_herb in herbs_set:
                            normalized_herbs.append(norm_herb)
                    
                    item['处方'] = normalized_herbs
        
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def load_task1_predictions(file_path):
    """加载任务1的预测结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 转换为字典，以便按ID查找
        result = {}
        for item in predictions:
            if 'ID' in item and '子任务1' in item:
                result[item['ID']] = item['子任务1']
        
        return result
    except Exception as e:
        print(f"Error loading task1 predictions: {e}")
        return {}

def create_herb_cooccurrence_matrix(data, herbs_list):
    """创建药物共现矩阵，分析药物搭配规律"""
    herb_to_idx = {herb: i for i, herb in enumerate(herbs_list)}
    idx_to_herb = {i: herb for i, herb in enumerate(herbs_list)}
    n_herbs = len(herbs_list)
    
    # 初始化共现矩阵
    cooccurrence = np.zeros((n_herbs, n_herbs))
    
    # 统计共现次数
    for item in data:
        if '处方' in item:
            if isinstance(item['处方'], str):
                try:
                    herbs = eval(item['处方'])
                except:
                    herbs = [item['处方']]
            else:
                herbs = item['处方']
            
            # 只考虑在药物列表中的药物
            herbs = [herb for herb in herbs if herb in herb_to_idx]
            
            # 更新共现矩阵
            for i, herb1 in enumerate(herbs):
                for herb2 in herbs[i:]:  # 包括自身
                    idx1 = herb_to_idx[herb1]
                    idx2 = herb_to_idx[herb2]
                    cooccurrence[idx1][idx2] += 1
                    if idx1 != idx2:
                        cooccurrence[idx2][idx1] += 1
    
    return cooccurrence, herb_to_idx, idx_to_herb

def create_syndrome_disease_herb_mapping(data, herbs_list):
    """创建证型-疾病-药物映射，分析不同证型和疾病的用药规律"""
    syndrome_herb_count = defaultdict(Counter)
    disease_herb_count = defaultdict(Counter)
    
    for item in data:
        if '证型' in item and '疾病' in item and '处方' in item:
            syndromes = item['证型'].split('|') if isinstance(item['证型'], str) else item['证型']
            disease = item['疾病']
            
            if isinstance(item['处方'], str):
                try:
                    herbs = eval(item['处方'])
                except:
                    herbs = [item['处方']]
            else:
                herbs = item['处方']
            
            # 过滤掉不在列表中的药物
            herbs = [herb for herb in herbs if herb in herbs_list]
            
            # 更新统计
            for syndrome in syndromes:
                syndrome_herb_count[syndrome].update(herbs)
            
            disease_herb_count[disease].update(herbs)
    
    # 转换为概率分布
    syndrome_herb_prob = {}
    for syndrome, counter in syndrome_herb_count.items():
        total = sum(counter.values())
        if total > 0:
            syndrome_herb_prob[syndrome] = {herb: count/total for herb, count in counter.items()}
    
    disease_herb_prob = {}
    for disease, counter in disease_herb_count.items():
        total = sum(counter.values())
        if total > 0:
            disease_herb_prob[disease] = {herb: count/total for herb, count in counter.items()}
    
    return syndrome_herb_prob, disease_herb_prob

def augment_data(data, herbs_list, cooccurrence=None, herb_to_idx=None, idx_to_herb=None, 
                syndrome_herb_prob=None, disease_herb_prob=None, augment_ratio=0.3):
    """数据增强：基于证型-药物和疾病-药物关系生成增强样本"""
    augmented_data = []
    
    for item in data:
        # 只增强有完整标签的数据
        if '证型' in item and '疾病' in item and '处方' in item:
            syndromes = item['证型'].split('|') if isinstance(item['证型'], str) else item['证型']
            disease = item['疾病']
            
            if isinstance(item['处方'], str):
                try:
                    herbs = eval(item['处方'])
                except:
                    herbs = [item['处方']]
            else:
                herbs = item['处方']
            
            # 过滤掉不在列表中的药物
            herbs = [herb for herb in herbs if herb in herbs_list]
            
            # 如果处方为空，跳过这个样本
            if not herbs:
                continue
                
            # 有一定概率进行增强
            if random.random() < augment_ratio:
                # 创建新样本
                new_item = item.copy()
                new_item['ID'] = f"{item['ID']}_aug"
                
                # 基于共现矩阵和证型疾病概率替换部分药物
                if cooccurrence is not None and herb_to_idx is not None and idx_to_herb is not None:
                    # 随机选择30%的药物进行替换，但不超过herbs的长度
                    replace_count = max(1, min(int(len(herbs) * 0.3), len(herbs)))
                    indices_to_replace = random.sample(range(len(herbs)), replace_count)
                    
                    new_herbs = herbs.copy()
                    for idx in indices_to_replace:
                        # 获取当前药物的共现药物
                        if herbs[idx] in herb_to_idx:
                            current_idx = herb_to_idx[herbs[idx]]
                            # 获取共现概率
                            probs = cooccurrence[current_idx]
                            # 选择不是当前处方的药物
                            candidates = [i for i in range(len(probs)) if idx_to_herb[i] not in herbs]
                            if candidates:
                                # 按共现概率选择
                                candidates_probs = probs[candidates]
                                # 避免零概率
                                candidates_probs = candidates_probs + 1e-10
                                candidates_probs = candidates_probs / candidates_probs.sum()
                                # 选择替换药物
                                replacement_idx = np.random.choice(candidates, p=candidates_probs)
                                new_herbs[idx] = idx_to_herb[replacement_idx]
                
                # 考虑证型和疾病的典型用药
                elif syndrome_herb_prob is not None and disease_herb_prob is not None:
                    # 建立候选药物池
                    candidates = set()
                    for syndrome in syndromes:
                        if syndrome in syndrome_herb_prob:
                            # 添加该证型常用药物
                            top_herbs = sorted(syndrome_herb_prob[syndrome].items(), 
                                              key=lambda x: x[1], reverse=True)[:20]
                            candidates.update([h for h, _ in top_herbs])
                    
                    if disease in disease_herb_prob:
                        # 添加该疾病常用药物
                        top_herbs = sorted(disease_herb_prob[disease].items(), 
                                          key=lambda x: x[1], reverse=True)[:20]
                        candidates.update([h for h, _ in top_herbs])
                    
                    # 移除已有药物
                    candidates = [h for h in candidates if h not in herbs]
                    
                    if candidates:
                        # 替换或添加药物
                        new_herbs = herbs.copy()
                        
                        # 随机替换30%药物，但不超过herbs的长度和候选药物的数量
                        replace_count = max(1, min(int(len(herbs) * 0.3), len(herbs)))
                        if len(new_herbs) > 0:
                            indices_to_replace = random.sample(range(len(new_herbs)), min(replace_count, len(new_herbs)))
                            for idx in indices_to_replace:
                                new_herbs[idx] = random.choice(candidates)
                        
                        # 控制在10-15味之间
                        target_count = random.randint(10, 15)
                        while len(new_herbs) < target_count and candidates:
                            new_herb = random.choice(candidates)
                            if new_herb not in new_herbs:
                                new_herbs.append(new_herb)
                                candidates.remove(new_herb)
                
                # 更新处方
                new_item['处方'] = new_herbs
                augmented_data.append(new_item)
    
    return augmented_data

def prepare_training_data(train_data, herbs_list, output_file, task1_predictions=None):
    """准备训练数据，将数据转换为提示-回答格式"""
    from prompts import generate_base_prompt, generate_few_shot_prompt
    
    logger = logging.getLogger(__name__)
    
    training_data = []
    skipped_items = 0
    empty_prescription_items = 0
    format_error_items = 0
    
    logger.info(f"Starting to prepare {len(train_data)} training samples")
    
    # 验证herbs_list
    if not herbs_list or len(herbs_list) == 0:
        logger.error("Herbs list is empty! Cannot process training data.")
        return []
    
    herbs_set = set(herbs_list)
    
    for i, item in enumerate(tqdm(train_data, desc="Preparing training data")):
        try:
            # 检查样本完整性
            if '处方' not in item:
                skipped_items += 1
                if i < 5:  # 只打印前几个错误的详细信息
                    logger.warning(f"Item {i} missing '处方' field: {item}")
                continue
                
            # 提取处方
            herbs = None
            if isinstance(item['处方'], str):
                try:
                    herbs = eval(item['处方'])
                except Exception as e:
                    format_error_items += 1
                    if i < 5:
                        logger.warning(f"Failed to parse prescription string for item {i}: {e}")
                        logger.warning(f"Prescription string: {item['处方']}")
                    herbs = [item['处方']]
            else:
                herbs = item['处方']
            
            # 过滤掉不在列表中的药物
            if herbs:
                original_count = len(herbs)
                herbs = [herb for herb in herbs if herb in herbs_set]
                if original_count > 0 and len(herbs) == 0:
                    logger.warning(f"Item {i}: All herbs filtered out! Original herbs: {item['处方']}")
            
            # 如果处方为空，跳过
            if not herbs:
                empty_prescription_items += 1
                if i < 5:
                    logger.warning(f"Item {i} has empty prescription after filtering: {item}")
                continue
            
            # 生成提示
            prompt = None
            try:
                # 50%使用少样本学习提示，50%使用基础提示
                if random.random() < 0.5:
                    prompt = generate_few_shot_prompt(item, herbs_list)
                else:
                    prompt = generate_base_prompt(item, herbs_list)
            except Exception as e:
                logger.error(f"Error generating prompt for item {i}: {e}")
                continue
                
            if not prompt:
                logger.warning(f"Empty prompt generated for item {i}")
                continue
            
            # 准备训练样本
            training_sample = {
                "input": prompt,
                "output": ", ".join(herbs)
            }
            
            training_data.append(training_sample)
            
            # 打印一些示例，帮助调试
            if i < 3:
                logger.info(f"Sample {i} - Input: {prompt[:100]}... Output: {training_sample['output']}")
                
        except Exception as e:
            logger.error(f"Error processing item {i}: {e}")
            if i < 5:
                logger.error(f"Problematic item: {item}")
    
    # 打印统计信息
    logger.info(f"Training data statistics:")
    logger.info(f"  - Total input items: {len(train_data)}")
    logger.info(f"  - Items without prescription field: {skipped_items}")
    logger.info(f"  - Items with empty prescriptions after filtering: {empty_prescription_items}")
    logger.info(f"  - Items with prescription format errors: {format_error_items}")
    logger.info(f"  - Final valid samples: {len(training_data)}")
    
    # 保存训练数据
    if len(training_data) > 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(training_data)} training samples to {output_file}")
    else:
        logger.error(f"No valid training samples to save!")
    
    return training_data

def prepare_evaluation_data(eval_data, herbs_list, output_file, task1_predictions=None):
    """准备评估数据，将数据转换为提示-回答格式"""
    from prompts import generate_base_prompt
    
    eval_samples = []
    
    for item in tqdm(eval_data, desc="Preparing evaluation data"):
        # 如果有任务1预测结果，获取预测的证型和疾病
        predicted_syndrome = None
        predicted_disease = None
        
        if task1_predictions and item['ID'] in task1_predictions:
            pred = task1_predictions[item['ID']]
            if len(pred) >= 2:
                predicted_syndrome = pred[0]
                predicted_disease = pred[1]
        
        # 生成提示
        prompt = generate_base_prompt(item, herbs_list, 
                                     predicted_syndromes=[predicted_syndrome] if predicted_syndrome else None,
                                     predicted_disease=predicted_disease)
        
        # 准备评估样本
        eval_sample = {
            "id": item['ID'],
            "input": prompt
        }
        
        # 如果有真实标签，添加到样本中
        if '处方' in item:
            if isinstance(item['处方'], str):
                try:
                    herbs = eval(item['处方'])
                except:
                    herbs = [item['处方']]
            else:
                herbs = item['处方']
            
            # 过滤掉不在列表中的药物
            herbs = [herb for herb in herbs if herb in herbs_list]
            
            if herbs:
                eval_sample["reference"] = ", ".join(herbs)
        
        eval_samples.append(eval_sample)
    
    # 保存评估数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in eval_samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(eval_samples)} evaluation samples to {output_file}")
    return eval_samples 