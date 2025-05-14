import logging
import os
import random
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 现在可以直接导入项目内的模块
from task1.model import TCMJointModel, dynamic_threshold_prediction
from utils.config import Config
from utils.data_utils import get_dataloaders

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    """设置随机种子，确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置cuDNN为确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    pretrained_model=None, 
    train_file=None, 
    dev_file=None, 
    output_dir=None, 
    max_seq_length=None, 
    train_batch_size=None, 
    eval_batch_size=None, 
    learning_rate=None, 
    weight_decay=None, 
    num_epochs=None, 
    warmup_ratio=None, 
    seed=None, 
    use_class_weights=True, 
    use_joint_modeling=True,
    val_ratio=0.2
):
    """训练模型的主函数，可直接调用"""
    # 使用参数覆盖默认配置
    config = Config()
    
    # 更新配置
    if pretrained_model:
        config.PRETRAINED_MODEL = pretrained_model
    if train_file:
        config.TRAIN_FILE = train_file
    if dev_file:
        config.DEV_FILE = dev_file
    if output_dir:
        config.OUTPUT_DIR = output_dir
    if max_seq_length:
        config.MAX_SEQ_LENGTH = max_seq_length
    if train_batch_size:
        config.TRAIN_BATCH_SIZE = train_batch_size
    if eval_batch_size:
        config.EVAL_BATCH_SIZE = eval_batch_size
    if learning_rate:
        config.LEARNING_RATE = learning_rate
    if weight_decay:
        config.WEIGHT_DECAY = weight_decay
    if num_epochs:
        config.NUM_EPOCHS = num_epochs
    if warmup_ratio:
        config.WARMUP_RATIO = warmup_ratio
    if seed:
        config.SEED = seed
    
    # 特殊标志
    config.USE_CLASS_WEIGHTS = use_class_weights
    config.USE_JOINT_MODELING = use_joint_modeling

    # 设置随机种子
    seed_everything(config.SEED)

    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 加载tokenizer
    logger.info(f"Loading tokenizer from {config.PRETRAINED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRETRAINED_MODEL,
        cache_dir=config.CACHE_DIR
    )

    # 获取数据加载器，划分训练集和验证集
    logger.info(f"Preparing datasets with validation ratio: {val_ratio}")
    train_dataloader, val_dataloader, _, syndrome_weights = get_dataloaders(
        tokenizer, config, val_ratio=val_ratio
    )

    # 初始化模型
    logger.info(f"Initializing model from {config.PRETRAINED_MODEL}")
    model = TCMJointModel(config)

    # 设置证型类权重
    if syndrome_weights is not None and config.USE_CLASS_WEIGHTS:
        model.set_syndrome_weights(syndrome_weights)

    # 将模型移动到GPU（如果可用）
    model = model.to(config.DEVICE)

    # 梯度累积步数
    gradient_accumulation_steps = 1

    # 优化器和调度器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)

    # 计算总训练步数
    num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    max_training_steps = config.NUM_EPOCHS * num_update_steps_per_epoch

    # 设置学习率调度器
    warmup_steps = int(max_training_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_training_steps
    )

    # 开始训练
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Validation examples = {len(val_dataloader.dataset)}")
    logger.info(f"  Num epochs = {config.NUM_EPOCHS}")
    logger.info(f"  Total optimization steps = {max_training_steps}")
    logger.info(f"  Warmup steps = {warmup_steps}")

    # 全局步数
    global_step = 0
    best_f1 = 0.0
    best_epoch = 0

    # 训练循环
    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        # 进度条
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            # 将批次数据移动到设备
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            syndrome_labels = batch['syndrome_labels'].to(config.DEVICE)
            disease_labels = batch['disease_labels'].to(config.DEVICE)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                syndrome_labels=syndrome_labels,
                disease_labels=disease_labels
            )

            loss = outputs['loss']

            # 如果使用梯度累积，需要除以累积步数
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # 累积损失
            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_steps += 1

            # 更新进度条
            progress_bar.set_postfix({"loss": epoch_loss / epoch_steps})

            # 梯度累积
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 保存检查点
                if global_step % config.SAVE_STEPS == 0:
                    # 验证模型
                    logger.info(f"Evaluating model at step {global_step}")
                    results = evaluate(model, val_dataloader, config)
                    syndrome_f1 = results['syndrome_f1']

                    # 如果结果更好，则保存模型
                    if syndrome_f1 > best_f1:
                        best_f1 = syndrome_f1
                        best_epoch = epoch

                        # 保存模型
                        checkpoint_dir = os.path.join(config.OUTPUT_DIR, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
                        logger.info(f"Model saved to {checkpoint_dir}")

                        # 保存评估结果
                        with open(os.path.join(checkpoint_dir, "results.txt"), "w") as f:
                            f.write(f"Step: {global_step}\n")
                            for key, value in results.items():
                                f.write(f"{key}: {value}\n")

        # 每个epoch结束后评估
        logger.info(f"Evaluating model after epoch {epoch + 1}")
        results = evaluate(model, val_dataloader, config)
        syndrome_f1 = results['syndrome_f1']

        logger.info(f"Epoch {epoch + 1} - Syndrome F1: {syndrome_f1:.4f}")

        # 如果结果更好，则保存模型
        if syndrome_f1 > best_f1:
            best_f1 = syndrome_f1
            best_epoch = epoch

            # 保存模型
            checkpoint_dir = os.path.join(config.OUTPUT_DIR, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
            logger.info(f"Model saved to {checkpoint_dir}")

            # 保存评估结果
            with open(os.path.join(checkpoint_dir, "results.txt"), "w") as f:
                f.write(f"Epoch: {epoch + 1}\n")
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")

    # 训练完成
    logger.info(f"Training completed. Best syndrome F1: {best_f1:.4f} at epoch {best_epoch + 1}")

    # 最后验证集上再次评估最佳模型
    logger.info("Final evaluation on validation set")
    best_checkpoint = os.path.join(config.OUTPUT_DIR, f"checkpoint-epoch-{best_epoch + 1}", "model.pt")
    model.load_state_dict(torch.load(best_checkpoint))
    final_results = evaluate(model, val_dataloader, config)

    logger.info("Final validation results:")
    for key, value in final_results.items():
        logger.info(f"  {key}: {value:.4f}")

    # 保存最终结果
    with open(os.path.join(config.OUTPUT_DIR, "final_results.txt"), "w") as f:
        f.write(f"Best epoch: {best_epoch + 1}\n")
        f.write("Final validation results:\n")
        for key, value in final_results.items():
            f.write(f"  {key}: {value:.4f}\n")

    return final_results


def evaluate(model, dataloader, config):
    """评估模型"""
    model.eval()

    all_syndrome_preds = []
    all_syndrome_labels = []
    all_disease_preds = []
    all_disease_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 将批次数据移动到设备
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            syndrome_labels = batch['syndrome_labels']
            disease_labels = batch['disease_labels']

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
            disease_labels_idx = torch.argmax(disease_labels, dim=1)

            # 收集预测和标签
            all_syndrome_preds.append(syndrome_preds.cpu())
            all_syndrome_labels.append(syndrome_labels)
            all_disease_preds.append(disease_preds.cpu())
            all_disease_labels.append(disease_labels_idx)

    # 连接所有批次的结果
    all_syndrome_preds = torch.cat(all_syndrome_preds, dim=0).numpy()
    all_syndrome_labels = torch.cat(all_syndrome_labels, dim=0).numpy()
    all_disease_preds = torch.cat(all_disease_preds, dim=0).numpy()
    all_disease_labels = torch.cat(all_disease_labels, dim=0).numpy()

    # 计算证型指标
    syndrome_f1 = f1_score(all_syndrome_labels, all_syndrome_preds, average='samples')
    syndrome_precision = precision_score(all_syndrome_labels, all_syndrome_preds, average='samples', zero_division=0)
    syndrome_recall = recall_score(all_syndrome_labels, all_syndrome_preds, average='samples', zero_division=0)

    # 计算疾病指标
    disease_accuracy = accuracy_score(all_disease_labels, all_disease_preds)

    # 计算总准确率 (子任务1评价指标)
    task1_acc = (syndrome_f1 + disease_accuracy) / 2

    # 收集结果
    results = {
        'syndrome_f1': syndrome_f1,
        'syndrome_precision': syndrome_precision,
        'syndrome_recall': syndrome_recall,
        'disease_accuracy': disease_accuracy,
        'task1_acc': task1_acc,
    }

    return results

# 直接运行时的默认参数
if __name__ == "__main__":
    # 可以根据需要修改这些参数
    params = {
        "pretrained_model": None,  # 预训练模型路径，默认使用config中的PRETRAINED_MODEL
        "train_file": None,  # 训练文件路径，默认使用config中的TRAIN_FILE
        "dev_file": None,  # 验证文件路径，默认使用config中的DEV_FILE
        "output_dir": None,  # 输出目录，默认使用config中的OUTPUT_DIR
        "max_seq_length": None,  # 最大序列长度，默认使用config中的MAX_SEQ_LENGTH
        "train_batch_size": None,  # 训练批次大小，默认使用config中的TRAIN_BATCH_SIZE
        "eval_batch_size": None,  # 评估批次大小，默认使用config中的EVAL_BATCH_SIZE
        "learning_rate": None,  # 学习率，默认使用config中的LEARNING_RATE
        "weight_decay": None,  # 权重衰减，默认使用config中的WEIGHT_DECAY
        "num_epochs": None,  # 训练轮数，默认使用config中的NUM_EPOCHS
        "warmup_ratio": None,  # 预热比例，默认使用config中的WARMUP_RATIO
        "seed": None,  # 随机种子，默认使用config中的SEED
        "use_class_weights": True,  # 是否使用类权重
        "use_joint_modeling": True,  # 是否使用联合建模
        "val_ratio": 0.2,  # 验证集占比
    }
    
    # 这里可以根据需要修改params中的值
    # 例如，设置预训练模型为roberta:
    # params["pretrained_model"] = "hfl/chinese-roberta-wwm-ext"
    # 或减少训练轮数：
    # params["num_epochs"] = 5
    
    # 运行训练
    train(**params)
