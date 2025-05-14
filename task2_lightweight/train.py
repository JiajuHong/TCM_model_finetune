#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轻量级中药处方推荐模型训练脚本"""

import os
import json
import argparse
import logging
import random
import time
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models import TCMHerbsLightModel, TCMHerbsWithSyndromeDiseaseModel
from utils import (
    load_herbs_list, 
    load_data,
    evaluate_herbs_prediction,
    process_herbs_prediction,
    load_task1_predictions
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[
        logging.FileHandler("task2_lightweight_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """配置类，管理训练参数"""
    def __init__(self):
        # 数据参数
        self.TRAIN_FILE = "../dataset/TCM-TBOSD-train.json"
        self.VAL_FILE = None
        self.HERBS_FILE = "../data/herbs.txt"
        self.TASK1_PREDICTIONS = "../task1/predictions.json"
        
        # 模型参数
        self.MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
        self.MAX_LENGTH = 512
        self.BATCH_SIZE = 16
        self.DROPOUT_RATE = 0.2
        
        # 训练参数
        self.LEARNING_RATE = 2e-5
        self.WEIGHT_DECAY = 0.01
        self.EPOCHS = 5
        self.WARMUP_RATIO = 0.1
        self.MAX_GRAD_NORM = 1.0
        self.SEED = 42
        
        # 输出参数
        self.OUTPUT_DIR = "./output"
        
        # 其他选项
        self.USE_TASK1_PREDS = False
        
        # 设备
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 保存步骤
        self.SAVE_STEPS = 100
        self.EVAL_STEPS = 100

def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, data_loader, device, mlb, threshold=0.5):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            herbs_labels = batch['herbs_labels'].to(device)
            ids = batch['ID']
            
            # 前向计算
            herbs_logits = model(input_ids, attention_mask)
            
            # 计算损失
            loss = criterion(herbs_logits, herbs_labels)
            total_loss += loss.item()
            
            # 保存预测和真实标签
            all_preds.append(herbs_logits.cpu())
            all_labels.append(herbs_labels.cpu())
            all_ids.extend(ids)
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    
    # 合并所有批次的预测和标签
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 二值化预测
    binary_preds = (torch.sigmoid(all_preds) > threshold).int().numpy()
    binary_labels = all_labels.int().numpy()
    
    # 计算指标
    metrics = {
        'loss': avg_loss,
        'micro_f1': f1_score(binary_labels, binary_preds, average='micro'),
        'macro_f1': f1_score(binary_labels, binary_preds, average='macro'),
        'micro_precision': precision_score(binary_labels, binary_preds, average='micro', zero_division=0),
        'micro_recall': recall_score(binary_labels, binary_preds, average='micro', zero_division=0),
        'micro_jaccard': jaccard_score(binary_labels, binary_preds, average='micro'),
        'samples_jaccard': jaccard_score(binary_labels, binary_preds, average='samples')
    }
    
    # 处理预测结果为中药列表
    herbs_predictions = process_herbs_prediction(all_preds, mlb, threshold)
    
    # 处理真实标签为中药列表
    herbs_labels = []
    for i in range(len(binary_labels)):
        herbs_labels.append(mlb.inverse_transform(binary_labels[i:i+1])[0])
    
    # 计算样本级别指标
    sample_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'jaccard': []
    }
    
    for pred, true in zip(herbs_predictions, herbs_labels):
        result = evaluate_herbs_prediction(pred, true)
        for key in sample_metrics:
            sample_metrics[key].append(result[key])
    
    # 添加样本级别平均指标
    for key, values in sample_metrics.items():
        metrics[f'sample_{key}'] = np.mean(values) if values else 0
    
    # 返回预测结果和评估指标
    prediction_results = {
        'ids': all_ids,
        'predictions': herbs_predictions,
        'true_labels': herbs_labels
    }
    
    return metrics, prediction_results

def plot_training_curves(metrics, output_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 8))
    
    # 绘制训练损失和验证损失
    epochs = range(1, len(metrics['train_loss']) + 1)
    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics['train_loss'], 'b-', marker='o', label='Training Loss')
    if 'val_loss' in metrics and len(metrics['val_loss']) > 0:
        plt.plot(epochs, metrics['val_loss'], 'r-', marker='s', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制F1分数和Jaccard分数
    plt.subplot(2, 1, 2)
    if 'micro_f1' in metrics and len(metrics['micro_f1']) > 0:
        plt.plot(epochs, metrics['micro_f1'], 'g-', marker='o', label='Micro F1')
    if 'sample_f1' in metrics and len(metrics['sample_f1']) > 0:
        plt.plot(epochs, metrics['sample_f1'], 'y-', marker='s', label='Sample F1')
    if 'micro_jaccard' in metrics and len(metrics['micro_jaccard']) > 0:
        plt.plot(epochs, metrics['micro_jaccard'], 'c-', marker='d', label='Micro Jaccard')
    if 'sample_jaccard' in metrics and len(metrics['sample_jaccard']) > 0:
        plt.plot(epochs, metrics['sample_jaccard'], 'm-', marker='x', label='Sample Jaccard')
    
    plt.title('Evaluation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def train_herbs_model(
    pretrained_model=None, 
    train_file=None, 
    val_file=None, 
    herbs_file=None,
    task1_predictions=None,
    output_dir=None, 
    max_length=None, 
    batch_size=None, 
    dropout_rate=None, 
    learning_rate=None, 
    weight_decay=None, 
    epochs=None, 
    warmup_ratio=None,
    max_grad_norm=None,
    seed=None, 
    use_task1_preds=None
):
    """训练中药处方推荐模型的主函数，可直接调用"""
    # 使用默认配置
    config = Config()
    
    # 使用传入参数覆盖默认配置
    if pretrained_model:
        config.MODEL_NAME = pretrained_model
    if train_file:
        config.TRAIN_FILE = train_file
    if val_file:
        config.VAL_FILE = val_file
    if herbs_file:
        config.HERBS_FILE = herbs_file
    if task1_predictions:
        config.TASK1_PREDICTIONS = task1_predictions
    if output_dir:
        config.OUTPUT_DIR = output_dir
    if max_length:
        config.MAX_LENGTH = max_length
    if batch_size:
        config.BATCH_SIZE = batch_size
    if dropout_rate:
        config.DROPOUT_RATE = dropout_rate
    if learning_rate:
        config.LEARNING_RATE = learning_rate
    if weight_decay:
        config.WEIGHT_DECAY = weight_decay
    if epochs:
        config.EPOCHS = epochs
    if warmup_ratio:
        config.WARMUP_RATIO = warmup_ratio
    if max_grad_norm:
        config.MAX_GRAD_NORM = max_grad_norm
    if seed:
        config.SEED = seed
    if use_task1_preds is not None:
        config.USE_TASK1_PREDS = use_task1_preds
    
    # 设置随机种子
    set_seed(config.SEED)
    
    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Using device: {config.DEVICE}")
    
    # 加载中药列表
    herbs_list = load_herbs_list(config.HERBS_FILE)
    logger.info(f"Loaded {len(herbs_list)} herbs")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    logger.info(f"Loaded tokenizer from {config.MODEL_NAME}")
    
    # 加载训练数据和验证数据
    train_loader, val_loader, mlb, complete_herbs_list = load_data(
        config.TRAIN_FILE,
        config.VAL_FILE,
        tokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.MAX_LENGTH,
        herbs_list=herbs_list
    )
    
    if not train_loader:
        logger.error("Failed to load training data")
        return
    
    # 使用从数据中提取的完整中药列表更新herbs_list
    if complete_herbs_list and len(complete_herbs_list) > len(herbs_list):
        logger.info(f"Updated herbs_list from {len(herbs_list)} to {len(complete_herbs_list)} items")
        herbs_list = complete_herbs_list
    
    logger.info(f"Train data loaded with {len(train_loader.dataset)} samples")
    if val_loader:
        logger.info(f"Validation data loaded with {len(val_loader.dataset)} samples")
    
    # 加载任务1预测结果(如果有)
    task1_predictions = None
    if config.USE_TASK1_PREDS and os.path.exists(config.TASK1_PREDICTIONS):
        task1_predictions = load_task1_predictions(config.TASK1_PREDICTIONS)
        logger.info(f"Loaded {len(task1_predictions)} task1 predictions")
    
    # 初始化模型
    if config.USE_TASK1_PREDS and task1_predictions:
        # 如果使用任务1的预测结果，加载增强模型
        num_syndromes = 10  # 证型数量，根据实际情况调整
        num_diseases = 4    # 疾病数量，根据实际情况调整
        
        model = TCMHerbsWithSyndromeDiseaseModel(
            model_name=config.MODEL_NAME,
            dropout_rate=config.DROPOUT_RATE,
            num_herbs=len(herbs_list),
            num_syndromes=num_syndromes,
            num_diseases=num_diseases
        )
        logger.info("Initialized model with syndrome and disease information integration")
    else:
        # 否则使用基本模型
        model = TCMHerbsLightModel(
            model_name=config.MODEL_NAME,
            dropout_rate=config.DROPOUT_RATE,
            num_herbs=len(herbs_list)
        )
        logger.info("Initialized basic herbs recommendation model")
    
    model = model.to(config.DEVICE)
    
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    
    # 优化器参数分组 - 遵循Transformers推荐的设置
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config.WEIGHT_DECAY
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.LEARNING_RATE
    )
    
    # 学习率调度器
    # 计算总训练步数
    gradient_accumulation_steps = 1  # 可以根据需要调整
    num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = config.EPOCHS * num_update_steps_per_epoch
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    # 记录训练指标
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'micro_f1': [],
        'micro_jaccard': [],
        'sample_f1': [],
        'sample_jaccard': []
    }
    
    # 最佳模型跟踪
    best_metric = -float('inf')  # 初始化为负无穷大，确保任何有效指标都会更新它
    best_epoch = 0
    best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    
    # 打印训练信息
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader.dataset)}")
    if val_loader:
        logger.info(f"  Num validation examples = {len(val_loader.dataset)}")
    logger.info(f"  Num epochs = {config.EPOCHS}")
    logger.info(f"  Total optimization steps = {total_steps}")
    logger.info(f"  Warmup steps = {int(total_steps * config.WARMUP_RATIO)}")
    
    # 全局步数
    global_step = 0
    
    # 训练循环
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        # 进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            # 准备数据
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            herbs_labels = batch['herbs_labels'].to(config.DEVICE)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向计算
            if config.USE_TASK1_PREDS and task1_predictions:
                # 需要添加证型和疾病信息的处理
                # 这里简化处理，实际应该根据batch的ID获取对应的证型疾病预测
                herbs_logits = model(input_ids, attention_mask)  # 简化版，忽略证型疾病输入
            else:
                herbs_logits = model(input_ids, attention_mask)
            
            # 计算损失
            loss = criterion(herbs_logits, herbs_labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            
            # 优化器步进
            optimizer.step()
            scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
            
            global_step += 1
            
            # 阶段性评估
            if global_step % config.EVAL_STEPS == 0 and val_loader:
                logger.info(f"Evaluating at step {global_step}...")
                val_metrics, _ = evaluate_model(model, val_loader, config.DEVICE, mlb)
                
                # 监控指标为样本级别的F1分数和Jaccard相似度的平均值
                current_metric = (val_metrics['sample_f1'] + val_metrics['sample_jaccard']) / 2
                
                logger.info(f"Step {global_step} - Val Loss: {val_metrics['loss']:.4f}, "
                          f"Sample F1: {val_metrics['sample_f1']:.4f}, "
                          f"Sample Jaccard: {val_metrics['sample_jaccard']:.4f}")
                
                # 如果性能提升，保存模型
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch + 1
                    
                    # 保存模型
                    checkpoint_dir = os.path.join(config.OUTPUT_DIR, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
                    logger.info(f"New best model saved to {checkpoint_dir} with metric: {best_metric:.4f}")
                
                # 回到训练模式
                model.train()
        
        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        
        # 保存每个周期的模型
        epoch_model_path = os.path.join(config.OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(epoch_model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(epoch_model_path, "model.pt"))
        
        # 验证
        if val_loader:
            logger.info(f"Evaluating after epoch {epoch+1}...")
            val_metrics, _ = evaluate_model(model, val_loader, config.DEVICE, mlb)
            
            # 记录验证指标
            metrics['val_loss'].append(val_metrics['loss'])
            metrics['micro_f1'].append(val_metrics['micro_f1'])
            metrics['micro_jaccard'].append(val_metrics['micro_jaccard'])
            metrics['sample_f1'].append(val_metrics['sample_f1'])
            metrics['sample_jaccard'].append(val_metrics['sample_jaccard'])
            
            # 监控指标为样本级别的F1分数和Jaccard相似度的平均值
            current_metric = (val_metrics['sample_f1'] + val_metrics['sample_jaccard']) / 2
            
            logger.info(
                f"Epoch {epoch+1}/{config.EPOCHS} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Micro F1: {val_metrics['micro_f1']:.4f}, "
                f"Sample F1: {val_metrics['sample_f1']:.4f}, "
                f"Sample Jaccard: {val_metrics['sample_jaccard']:.4f}"
            )
            
            # 如果性能提升，保存模型
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1
                best_model_path = os.path.join(epoch_model_path, "model.pt")
                logger.info(f"New best model at epoch {best_epoch} with metric: {best_metric:.4f}")
        else:
            # 如果没有验证集，则使用训练损失作为指标
            # 损失越小越好，所以使用负值进行比较
            current_metric = -avg_train_loss
            
            logger.info(
                f"Epoch {epoch+1}/{config.EPOCHS} - "
                f"Train Loss: {avg_train_loss:.4f}"
            )
            
            # 如果性能提升，保存模型
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1
                best_model_path = os.path.join(epoch_model_path, "model.pt")
                logger.info(f"New best model at epoch {best_epoch} with train loss: {avg_train_loss:.4f}")
        
        # 计算epoch耗时
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
    
    # 绘制训练曲线
    plot_training_curves(metrics, config.OUTPUT_DIR)
    
    # 保存MultiLabelBinarizer
    mlb_path = os.path.join(config.OUTPUT_DIR, 'mlb.json')
    with open(mlb_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': mlb.classes_.tolist()}, f, ensure_ascii=False, indent=2)
    
    # 保存herbs_list
    herbs_list_path = os.path.join(config.OUTPUT_DIR, 'herbs_list.json')
    with open(herbs_list_path, 'w', encoding='utf-8') as f:
        json.dump(herbs_list, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Training completed. Best model (epoch {best_epoch}) saved with metric: {best_metric:.4f}")
    
    # 复制最佳模型到标准位置
    if best_epoch > 0 and os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
        logger.info(f"Best model copied to {os.path.join(config.OUTPUT_DIR, 'best_model.pth')}")
    else:
        # 如果没有找到最佳模型，使用最后一个epoch的模型
        last_epoch_path = os.path.join(config.OUTPUT_DIR, f"checkpoint-epoch-{config.EPOCHS}", "model.pt")
        if os.path.exists(last_epoch_path):
            import shutil
            shutil.copy(last_epoch_path, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
            logger.info(f"No best model found. Last epoch model copied to {os.path.join(config.OUTPUT_DIR, 'best_model.pth')}")
    
    # 保存最终评估结果
    with open(os.path.join(config.OUTPUT_DIR, "final_results.txt"), "w") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        if val_loader and len(metrics['val_loss']) > 0:
            best_idx = best_epoch - 1  # 转换为索引
            if 0 <= best_idx < len(metrics['val_loss']):
                f.write(f"Best validation loss: {metrics['val_loss'][best_idx]:.4f}\n")
                f.write(f"Best sample F1: {metrics['sample_f1'][best_idx]:.4f}\n")
                f.write(f"Best sample Jaccard: {metrics['sample_jaccard'][best_idx]:.4f}\n")
        else:
            # 如果没有验证数据，记录最佳训练损失
            if best_epoch > 0 and best_epoch <= len(metrics['train_loss']):
                best_idx = best_epoch - 1
                f.write(f"Best train loss: {metrics['train_loss'][best_idx]:.4f}\n")
                f.write(f"No validation data was provided.\n")
    
    return best_model_path, mlb

def main():
    parser = argparse.ArgumentParser(description="Train TCM herbs recommendation model")
    
    # 数据参数
    parser.add_argument("--train_file", type=str, default="../dataset/TCM-TBOSD-train.json", help="Path to training data file")
    parser.add_argument("--val_file", type=str, default=None, help="Path to validation data file")
    parser.add_argument("--herbs_file", type=str, default="../data/herbs.txt", help="Path to herbs list file")
    parser.add_argument("--task1_predictions", type=str, default="../task1/predictions.json", help="Path to task1 predictions")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext", help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    
    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    # 其他选项
    parser.add_argument("--use_task1_preds", action="store_true", help="Use task1 predictions")
    
    args = parser.parse_args()
    
    # 调用训练函数
    train_herbs_model(
        pretrained_model=args.model_name,
        train_file=args.train_file,
        val_file=args.val_file,
        herbs_file=args.herbs_file,
        task1_predictions=args.task1_predictions,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        use_task1_preds=args.use_task1_preds
    )

if __name__ == "__main__":
    main() 