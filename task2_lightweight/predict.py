#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轻量级中药处方推荐模型预测脚本"""

import os
import json
import argparse
import logging
from tqdm import tqdm
import torch
import numpy as np
from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

from models import TCMHerbsLightModel, TCMHerbsWithSyndromeDiseaseModel
from utils import extract_text_from_case, process_herbs_prediction, load_task1_predictions

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("task2_lightweight_predict.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_mlb(mlb_path):
    """加载MultiLabelBinarizer"""
    try:
        with open(mlb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        mlb = MultiLabelBinarizer()
        mlb.classes_ = np.array(data['classes'])
        logger.info(f"Loaded MLBinarizer with {len(mlb.classes_)} classes")
        return mlb
    except Exception as e:
        logger.error(f"Error loading MLBinarizer: {e}")
        return None

def load_herbs_json(herbs_list_path):
    """从JSON文件加载中药列表"""
    try:
        with open(herbs_list_path, 'r', encoding='utf-8') as f:
            herbs_list = json.load(f)
        logger.info(f"Loaded {len(herbs_list)} herbs from {herbs_list_path}")
        return herbs_list
    except Exception as e:
        logger.error(f"Error loading herbs list from JSON: {e}")
        return []

def load_model(model_path, model_type, model_name, num_herbs, num_syndromes=10, num_diseases=4):
    """加载训练好的模型"""
    try:
        if model_type == "basic":
            model = TCMHerbsLightModel(
                model_name=model_name,
                num_herbs=num_herbs
            )
        else:
            model = TCMHerbsWithSyndromeDiseaseModel(
                model_name=model_name,
                num_herbs=num_herbs,
                num_syndromes=num_syndromes,
                num_diseases=num_diseases
            )
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def predict(args):
    """使用训练好的模型进行预测"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 加载中药列表 - 直接从JSON加载
    try:
        herbs_list = load_herbs_json(args.herbs_list)
        logger.info(f"Loaded {len(herbs_list)} herbs")
    except Exception as e:
        logger.error(f"Error loading herbs list: {e}")
        return
    
    # 加载MultiLabelBinarizer
    mlb = load_mlb(args.mlb_file)
    if not mlb:
        logger.error("Failed to load MultiLabelBinarizer")
        return
    
    # 确保herbs_list和mlb.classes_一致
    if len(herbs_list) != len(mlb.classes_):
        logger.warning(f"Herbs list length ({len(herbs_list)}) differs from MLBinarizer classes ({len(mlb.classes_)})")
        logger.warning("Using MLBinarizer classes for consistency with training")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    logger.info(f"Loaded tokenizer from {args.model_name}")
    
    # 加载模型
    model = load_model(
        args.model_path, 
        args.model_type, 
        args.model_name, 
        num_herbs=len(mlb.classes_)  # 使用MLBinarizer中的类别数量
    )
    if not model:
        logger.error("Failed to load model")
        return
    model.to(device)
    model.eval()
    
    # 加载任务1预测结果(如果使用)
    task1_predictions = None
    if args.use_task1_preds and os.path.exists(args.task1_predictions):
        task1_predictions = load_task1_predictions(args.task1_predictions)
        logger.info(f"Loaded {len(task1_predictions)} task1 predictions")
    
    # 加载测试数据
    try:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        logger.info(f"Loaded {len(test_data)} test cases")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    # 预测结果
    predictions = {}
    
    # 遍历测试数据进行预测
    for case in tqdm(test_data, desc="Predicting"):
        case_id = case.get('ID', '')
        if not case_id:
            logger.warning("Case ID is missing, skipping")
            continue
        
        # 提取文本
        text = extract_text_from_case(case)
        
        # 使用tokenizer处理文本
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 将数据移动到设备上
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 准备证型疾病信息(如果使用)
        syndrome_probs = None
        disease_probs = None
        
        if args.use_task1_preds and task1_predictions and case_id in task1_predictions:
            pred_task1 = task1_predictions[case_id]
            # 这里需要将证型和疾病转换为概率向量，为简化处理，这里仅做示意
            # 实际应用中应该使用完整的转换逻辑
            if isinstance(pred_task1, list) and len(pred_task1) >= 2:
                syndrome = pred_task1[0]
                disease = pred_task1[1]
                # 简化处理，实际应该转换为独热编码或概率分布
                logger.info(f"Using task1 prediction for case {case_id}: {syndrome}, {disease}")
        
        with torch.no_grad():
            # 根据模型类型进行预测
            if args.model_type == "enhanced" and syndrome_probs is not None and disease_probs is not None:
                herbs_logits = model(input_ids, attention_mask, syndrome_probs, disease_probs)
            else:
                herbs_logits = model(input_ids, attention_mask)
        
        # 处理预测结果
        herbs_predictions = process_herbs_prediction(
            herbs_logits, 
            mlb, 
            threshold=args.threshold, 
            top_k=args.top_k
        )
        
        # 保存预测结果
        predictions[case_id] = herbs_predictions[0]  # 只有一个样本
    
    # 保存预测结果
    output = []
    for case_id, herbs in predictions.items():
        output.append({
            "ID": case_id,
            "子任务2": herbs
        })
    
    # 如果需要合并任务1的预测结果
    if args.merge_task1 and task1_predictions:
        # 合并任务1和任务2的预测结果
        final_output = []
        for item in output:
            case_id = item["ID"]
            merged_item = {"ID": case_id}
            
            # 添加任务1预测
            if case_id in task1_predictions:
                merged_item["子任务1"] = task1_predictions[case_id]
            
            # 添加任务2预测
            merged_item["子任务2"] = item["子任务2"]
            
            final_output.append(merged_item)
        
        output = final_output
    
    # 写入文件
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Predictions saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Predict TCM herbs recommendation")
    
    # 数据参数
    parser.add_argument("--test_file", type=str, default="../dataset/TCM-TBOSD-test-B.json", help="Path to test data file")
    parser.add_argument("--herbs_list", type=str, default="./output/herbs_list.json", help="Path to herbs list JSON file")
    parser.add_argument("--mlb_file", type=str, default="./output/mlb.json", help="Path to MLBinarizer JSON file")
    parser.add_argument("--task1_predictions", type=str, default='../task1/predictions.json', help="Path to task1 predictions")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="./output/best_model.pth", help="Path to trained model")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext", help="Base pretrained model name")
    parser.add_argument("--model_type", choices=["basic", "enhanced"], default="basic", help="Model type")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    # 预测参数
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    parser.add_argument("--top_k", type=int, default=15, help="Number of top herbs to select")
    
    # 输出参数
    parser.add_argument("--output_file", type=str, default="./predictions.json", help="Output prediction file")
    
    # 其他选项
    parser.add_argument("--use_task1_preds", action="store_true", help="Use task1 predictions")
    parser.add_argument("--merge_task1", action="store_true", help="Merge task1 predictions in output")
    
    args = parser.parse_args()
    
    predict(args)

if __name__ == "__main__":
    main() 