import torch
import os
import sys
import logging
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 现在可以直接导入项目内的模块
from task1.model import TCMJointModel, dynamic_threshold_prediction
from utils.config import Config
from utils.data_utils import TCMDataset, load_test_dataloader

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def predict(model, dataset, config, is_test=False):
    """生成预测结果"""
    model.eval()
    device = config.DEVICE
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
    )
    
    # 记录预测结果
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            # 提取样本ID
            ids = batch['id']
            
            # 将批次数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            # 证型预测
            syndrome_logits = outputs['syndrome_logits']
            syndrome_preds = dynamic_threshold_prediction(
                syndrome_logits, 
                min_labels=config.MIN_SYNDROME_LABELS, 
                max_labels=config.MAX_SYNDROME_LABELS
            )
            
            # 疾病预测
            disease_logits = outputs['disease_logits']
            disease_preds = torch.argmax(disease_logits, dim=1)
            
            # 将索引转换为标签
            for i in range(len(ids)):
                sample_id = ids[i]
                
                # 获取预测的证型
                pred_syndrome_indices = torch.where(syndrome_preds[i])[0].cpu().numpy()
                pred_syndromes = [config.SYNDROME_LABELS[idx] for idx in pred_syndrome_indices]
                
                # 获取预测的疾病
                pred_disease_idx = disease_preds[i].item()
                pred_disease = config.DISEASE_LABELS[pred_disease_idx]
                
                # 将多个证型用|连接，然后与疾病组成列表，符合提交格式要求
                syndrome_str = "|".join(pred_syndromes)
                combined_result = [syndrome_str, pred_disease]
                
                # 保存预测结果
                result = {
                    "ID": sample_id,
                    "子任务1": combined_result
                }
                
                # 如果不是测试集，则添加真实标签(如果有的话)
                if not is_test and 'raw_syndromes' in batch and 'raw_disease' in batch:
                    raw_syndromes = batch['raw_syndromes']
                    raw_diseases = batch['raw_disease']
                    result["真实证型"] = raw_syndromes[i]
                    result["真实疾病"] = raw_diseases[i]
                
                results.append(result)
    
    return results

def evaluate(
    model_path, 
    test_file=None, 
    output_file="predictions.json", 
    pretrained_model=None,
    max_seq_length=None,
    eval_batch_size=None
):
    """评估函数，可直接调用"""
    # 使用参数覆盖默认配置
    config = Config()
    if pretrained_model:
        config.PRETRAINED_MODEL = pretrained_model
    if max_seq_length:
        config.MAX_SEQ_LENGTH = max_seq_length
    if eval_batch_size:
        config.EVAL_BATCH_SIZE = eval_batch_size
    
    # 设置文件路径
    if test_file is None:
        test_file = config.DEV_FILE
    
    # 加载tokenizer
    logger.info(f"Loading tokenizer from {config.PRETRAINED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRETRAINED_MODEL,
        cache_dir=config.CACHE_DIR
    )
    
    # 加载测试数据
    logger.info(f"Loading test data from {test_file}")
    if test_file == config.DEV_FILE:
        # 使用专门的函数加载测试数据
        test_dataloader = load_test_dataloader(tokenizer, config)
        test_dataset = test_dataloader.dataset
    else:
        # 自定义测试文件的情况
        test_dataset = TCMDataset(test_file, tokenizer, config, is_test=True)
    
    # 初始化模型
    logger.info("Initializing model")
    model = TCMJointModel(config)
    
    # 加载预训练权重
    logger.info(f"Loading trained weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    
    # 生成预测
    logger.info("生成证型和疾病预测...")
    predictions = predict(model, test_dataset, config, is_test=True)
    
    # 保存原始预测结果
    logger.info(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    
    logger.info("预测完成。由于测试数据没有标签，无法计算准确率。")
    return None
    
# 直接运行时的默认参数
if __name__ == "__main__":
    # 可以根据需要修改这些参数
    params = {
        "model_path": "../models/finetuned/checkpoint-epoch-1/model.pt",  # 模型路径
        "test_file": None,  # 测试文件路径，默认使用config中的DEV_FILE
        "output_file": "predictions.json",  # 输出预测文件路径
        "pretrained_model": None,  # 预训练模型路径，默认使用config中的PRETRAINED_MODEL
        "max_seq_length": None,  # 最大序列长度，默认使用config中的MAX_SEQ_LENGTH
        "eval_batch_size": None,  # 评估批次大小，默认使用config中的EVAL_BATCH_SIZE
    }
    
    # 这里可以根据需要修改params中的值
    
    # 运行评估
    evaluate(**params) 