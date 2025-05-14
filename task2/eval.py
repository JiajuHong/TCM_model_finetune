"""中药处方推荐模型评估脚本"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import Config
from data_utils import load_herbs_list, load_dataset, prepare_evaluation_data
from utils import (
    normalize_herbs,
    postprocess_herbs,
    evaluate_prescription,
    select_best_candidate
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("task2_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path, base_model_path=None):
    """加载模型和分词器"""
    logger.info(f"Loading model from {model_path}")
    
    if base_model_path:
        # 如果有基础模型路径，说明是PEFT模型
        logger.info(f"Using base model {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            device_map="auto"
        )
    else:
        # 如果没有基础模型路径，直接加载
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
    
    # 特殊标记处理
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def generate_prescription(model, tokenizer, prompt, config):
    """生成中药处方"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    
    # 使用多个候选生成
    candidates = []
    for _ in range(config.num_candidates):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=200,
                temperature=config.temperature,
                top_p=config.top_p,
                num_beams=config.num_beams,
                do_sample=True if config.temperature > 0 else False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        candidates.append(generated_text.strip())
    
    return candidates

def evaluate_model(config, model_path=None, data_path=None, herbs_file=None, base_model_path=None):
    """评估中药处方推荐模型"""
    # 使用传入的路径或配置文件中的路径
    model_path = model_path or config.output_dir
    data_path = data_path or config.val_file
    herbs_file = herbs_file or config.herbs_file
    base_model_path = base_model_path or config.base_model
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path, base_model_path)
    
    # 加载中药列表
    herbs_list_path = os.path.join(model_path, "herbs_list.json")
    if os.path.exists(herbs_list_path):
        logger.info(f"Loading herbs list from {herbs_list_path}")
        with open(herbs_list_path, "r", encoding="utf-8") as f:
            herbs_list = json.load(f)
    else:
        logger.info(f"Loading herbs list from {herbs_file}")
        herbs_list = load_herbs_list(herbs_file)
    
    herbs_set = set(herbs_list)
    
    # 加载验证数据
    logger.info(f"Loading evaluation data from {data_path}")
    eval_data = load_dataset(data_path, herbs_list)
    
    # 准备评估数据
    eval_output_file = os.path.join(os.path.dirname(model_path), "eval_data.jsonl")
    evaluation_data = prepare_evaluation_data(eval_data, herbs_list, eval_output_file)
    
    # 评估结果
    metrics_list = []
    predictions = {}
    
    # 生成处方并评估
    logger.info("Generating and evaluating prescriptions")
    for item in tqdm(evaluation_data, desc="Evaluating"):
        sample_id = item["id"]
        prompt = item["input"]
        
        # 生成候选处方
        candidates = generate_prescription(model, tokenizer, prompt, config)
        
        # 后处理并选择最佳候选
        reference = item.get("reference", None)
        best_candidate = select_best_candidate(
            candidates, 
            reference, 
            herbs_set,
            config.min_herbs,
            config.max_herbs
        )
        
        # 保存预测结果
        predictions[sample_id] = best_candidate
        
        # 如果有参考处方，计算指标
        if reference:
            ref_herbs = normalize_herbs(reference)
            if ref_herbs:
                metrics = evaluate_prescription(ref_herbs, best_candidate)
                metrics["sample_id"] = sample_id
                metrics_list.append(metrics)
    
    # 计算平均指标
    if metrics_list:
        avg_metrics = {
            "jaccard": np.mean([m["jaccard"] for m in metrics_list]),
            "recall": np.mean([m["recall"] for m in metrics_list]),
            "precision": np.mean([m["precision"] for m in metrics_list]),
            "f1": np.mean([m["f1"] for m in metrics_list]),
            "avg_herbs": np.mean([m["avg_herbs"] for m in metrics_list]),
            "task2_score": np.mean([m["task2_score"] for m in metrics_list])
        }
        
        logger.info("Evaluation metrics:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    
    # 保存评估结果
    results_file = os.path.join(os.path.dirname(model_path), "eval_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": avg_metrics if metrics_list else {},
            "detailed_metrics": metrics_list,
            "predictions": {id_: herbs for id_, herbs in predictions.items()}
        }, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Evaluation results saved to {results_file}")
    
    return avg_metrics if metrics_list else {}, predictions

if __name__ == "__main__":
    config = Config()
    evaluate_model(config) 