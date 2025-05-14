#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中药处方推荐模型训练脚本"""

import json
import logging
import os
import random
import warnings

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)

from config import Config
from data_utils import (
    load_herbs_list,
    load_dataset,
    load_task1_predictions,
    create_herb_cooccurrence_matrix,
    create_syndrome_disease_herb_mapping,
    augment_data,
    prepare_training_data,
    prepare_evaluation_data
)

warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("task2_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_random_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

def load_tokenizer_and_model(config):
    """加载分词器和模型"""
    logger.info(f"Loading tokenizer from {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 特殊标记处理 - 修复pad_token问题
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token for Qwen model")
        # Qwen模型不支持添加新的特殊token，使用eos_token作为pad_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Using eos_token as pad_token with ID: {tokenizer.pad_token_id}")
    else:
        logger.info(f"Tokenizer already has pad_token: {tokenizer.pad_token}, id: {tokenizer.pad_token_id}")
        
    # 确保模型知道pad_token_id
    logger.info(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    
    # 量化配置
    if config.load_in_4bit:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        quantization_config = None
    
    # 加载模型
    logger.info(f"Loading model from {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False  # 禁用KV缓存以避免与梯度检查点的冲突
    )
    
    # 设置模型的pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Set model.config.pad_token_id to {model.config.pad_token_id}")
    
    # 为量化训练准备模型
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # 启用梯度检查点以节省内存
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")
    
    # 配置LoRA
    logger.info(f"Configuring LoRA with rank={config.lora_r}, alpha={config.lora_alpha}")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 获取PEFT模型
    model = get_peft_model(model, lora_config)
    logger.info(f"Model loaded and configured with PEFT")
    
    return tokenizer, model

def preprocess_training_data(config):
    """处理训练数据，包括数据增强和格式准备"""
    # 检查并创建药物列表
    if not os.path.exists(config.herbs_file):
        logger.warning(f"Herbs file {config.herbs_file} not found. Creating from default herbs list.")
        herbs_list = ['冬瓜皮', '沉香', '茜草炭', '浮小麦', '炙甘草', '炒白扁豆', '砂仁', '合欢花', '北刘寄奴', '炒六神曲', '炒决明子', '益母草', '酒苁蓉', '炒僵蚕', '稀莶草', '秦艽', '黄酒', '瞿麦', ' 白鲜皮', '熟地黄', '扁蓄', '诃子肉', '煅牡蛎', '鸡血藤', '党参', '瓜蒌', '莲子', '酒五味子', '金钱草', '法半夏', '北败酱草', '花椒', '吴茱萸(粉)', '桑白皮', '茯神', '桂枝', '降香', '制远志', '琥珀', '佛手', '麦芽', '水红花子', '金银花', '马鞭草', '半枝莲', '炮姜', '生酸枣仁', '盐补骨脂', '炒瓜蒌子', '珍珠母', '乌药', '茵陈', '地肤子', '酸枣仁', '槟榔', '大青叶', '人参片', '麸煨肉豆蔻', '蛤蚧', '路路通', '蝉蜕', '马勃', '香橼', '络石藤', '狗脊', '蜈蚣', '制川乌', '白扁豆花', '麻黄', '射干', '厚朴', '蜂蜜', '柏子仁', '炒谷芽', '蜜百合', ' 石菖蒲', '白薇', '续断', '炒川楝子', '黄连片', '绵萆薢', '鹿角胶', '翻白草', '羚羊角粉', '天麻', '山慈菇', '菊花', '炒芥子', '墨旱莲', '蜜枇杷叶', '川芎', '酒大黄', '焦山楂', '红曲', '山药', '牡蛎', '海藻', '夏枯草', '白前', '白芍', '茯苓皮', '煅自然铜', '附片 ', '土茯苓', '制何首乌', '炒莱菔子', '黄芩', '蒲黄', '紫石英', '透骨草', '绞股蓝', '泽泻', '甘松', ' 炒酸枣仁', '儿茶', '马齿苋', '太子参', '薏苡仁', '萹蓄', '青蒿', '苏木', '桑叶', '连翘', '穿山龙', '忍冬藤', '苦参', '炒茺蔚子', '防己', '益母草炭', '莲须', '猫眼草', '麸炒芡实', ' 炒牛蒡子', '龟甲胶', '蜜槐角', '柿蒂', '龙骨', '泽兰', '桔梗', '青葙子', '冰片', '大枣', '侧柏叶', '三七粉', '醋乳香', '川牛膝', '全蝎', '合欢皮', '首乌藤', '醋鳖甲', '炒蔓荆子', ' 烫骨碎补', '紫苏叶', '盐沙苑子', '南沙参', '石见穿', '胆南星', '焦白术', '酒黄芩', '白术', '鬼箭羽', '玫瑰花', '干姜', '牡丹皮', '白花蛇舌草', '酒当归', '火麻仁', '炒桃仁', '醋鸡内金', '磁石', '醋龟甲', '白茅根', '肉桂', '白及', '油松节', '炒苍耳子', '化橘红', '佩兰', '芦根', '紫草', '酒萸肉', '丹参', '柴胡', '制巴戟天', '木蝴蝶', '炒紫苏子', '浮萍', '栀子', '甘草片', '木香', '丝瓜络', '炒麦芽', '板蓝根', '车前草', '炒王不留行', '朱砂', '醋三棱', '辛夷', '土鳖虫', '煅龙骨', '炒白芍', '炒白果仁', '芒硝', '赭石', '西洋参', '桑枝', '红景天', '锁阳', '淫羊藿', '酒乌梢蛇', '制草乌', '肉苁蓉片', '麸炒枳壳', '炒苦杏仁', '炙黄芪', '黄连', '重楼', '细辛', '蜜旋覆花', '醋没药', '玉竹', '蛤壳', '草豆蔻', '炙淫羊藿', '广藿香', '麸炒枳实', '鱼腥草', '鹿角霜', '通草', '烫水蛭', '水牛角', '烫狗脊', '盐续断', '盐益智仁', '常山', '百部', '阿胶', '藁本片', '制吴茱萸', '豆蔻', '酒女贞子', '片姜黄', '蜜款冬花', '龙胆', '寒水石', '莲子心', '荷叶', '防风', '炒蒺藜', '川贝母', '虎杖', '海桐皮', '甘草', '赤石脂', '麻黄根', '郁金', '海风藤', '青皮', '地龙', '地榆', '石韦', '焦栀子', '盐杜仲', '清半夏', '盐知母', '薤白', '茜草', '荆芥炭', '百合', '龙齿', '石决明', '炒葶苈子', '知母', '赤小豆', '麸炒白术', '酒仙茅', '淡竹叶', '大黄', '海螵蛸', '仙鹤草', '白芷', '麸炒薏苡仁', '青风藤', '前胡', '升麻', '海浮石', '制天南星', '麸炒山药', '蒲公英', '豨莶草', '当归', '醋莪术', '薄荷', '红参片', '生地黄', '苦地丁', '炒槐米', '蜜桑白皮', '盐小茴香', '麸炒苍 术', '姜半夏', '钟乳石', '桑椹', '瓜蒌皮', '葛根', '桑螵蛸', '浙贝片', '菟丝子', '醋延胡索', '艾叶', '五加皮', '炒冬瓜子', '瓦楞子', '盐黄柏', '醋五灵脂', '石膏', '醋山甲', '檀香', '皂角刺', '红花', '野菊花', '木瓜', '蜜麻黄', '槲寄生', '密蒙花', '蜜百部', '蜜紫菀', '茯苓', '海金沙', '麦冬', '猪苓', '天竺黄', '石斛', '枸杞子', '徐长卿', '醋香附', '麸神曲', '黄芪', '郁李仁', '枯矾', '盐车前子', '伸筋草', '草果仁', '山楂', '炒稻芽', '威灵仙', '淡豆豉', '蛇莓', '丁香', '盐荔枝核', '绵马贯众', '黄柏', '独活', '覆盆子', '龙眼肉', '老鹳草', ' 乌梅', '紫苏梗', '制白附子', '大腹皮', '竹茹', '天花粉', '乌梅炭', '滑石粉', '冬葵子', '灯心草', '六月雪', '牛膝', '陈皮', '荆芥', '炒甘草', '北沙参', '地骷髅', '地骨皮', '赤芍', ' 玄参', '桑葚', '酒黄精', '羌活', '钩藤', '天冬']
        # 清理药物名称中的空格
        herbs_list = [herb.strip() for herb in herbs_list]
        # 确保目录存在
        os.makedirs(os.path.dirname(config.herbs_file), exist_ok=True)
        # 写入文件
        with open(config.herbs_file, 'w', encoding='utf-8') as f:
            for herb in herbs_list:
                f.write(f"{herb}\n")
        logger.info(f"Created herbs file with {len(herbs_list)} herbs")
    
    # 加载中药列表
    logger.info(f"Loading herbs list from {config.herbs_file}")
    herbs_list = load_herbs_list(config.herbs_file)
    herbs_set = set(herbs_list)
    logger.info(f"Loaded {len(herbs_list)} herbs")
    
    # 加载训练数据
    logger.info(f"Loading training data from {config.train_file}")
    all_data = load_dataset(config.train_file, herbs_list)
    logger.info(f"Loaded {len(all_data)} total samples")
    
    if len(all_data) == 0:
        logger.error(f"Failed to load any data from {config.train_file}. Please check file path and format.")
        raise ValueError(f"No data loaded from {config.train_file}")
    
    # 划分训练集和验证集
    logger.info(f"Splitting data with validation ratio {config.val_ratio}")
    random.shuffle(all_data)
    split_idx = int(len(all_data) * (1 - config.val_ratio))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # 验证处方数据的完整性
    valid_train_samples = [item for item in train_data if '处方' in item and isinstance(item.get('处方', []), list) and len(item.get('处方', [])) > 0]
    logger.info(f"Training samples with valid prescriptions: {len(valid_train_samples)} out of {len(train_data)}")
    
    if len(valid_train_samples) == 0:
        logger.warning("No training samples with valid prescriptions! Using all training data regardless of prescription validity.")
    else:
        train_data = valid_train_samples
    
    # 创建药物共现矩阵
    logger.info("Creating herb co-occurrence matrix")
    cooccurrence, herb_to_idx, idx_to_herb = create_herb_cooccurrence_matrix(train_data, herbs_list)
    
    # 创建证型和疾病的药物映射
    logger.info("Creating syndrome-disease-herb mappings")
    syndrome_herb_prob, disease_herb_prob = create_syndrome_disease_herb_mapping(train_data, herbs_list)
    
    # 数据增强
    logger.info("Augmenting training data")
    augmented_data = augment_data(
        train_data, 
        herbs_list,
        cooccurrence, 
        herb_to_idx, 
        idx_to_herb,
        syndrome_herb_prob, 
        disease_herb_prob
    )
    logger.info(f"Generated {len(augmented_data)} augmented samples")
    
    # 合并原始数据和增强数据
    all_train_data = train_data + augmented_data
    logger.info(f"Total training samples after augmentation: {len(all_train_data)}")
    
    # 尝试加载任务1预测结果（如果有）
    task1_predictions = None
    if os.path.exists(config.task1_predictions):
        logger.info(f"Loading task1 predictions from {config.task1_predictions}")
        task1_predictions = load_task1_predictions(config.task1_predictions)
        logger.info(f"Loaded {len(task1_predictions)} task1 predictions")
    
    # 准备训练数据
    train_output_file = os.path.join(os.path.dirname(config.output_dir), "train_data.jsonl")
    logger.info(f"Preparing training data and saving to {train_output_file}")
    training_data = prepare_training_data(all_train_data, herbs_list, train_output_file, task1_predictions)
    
    if len(training_data) == 0:
        logger.error("No training samples were generated! Check data format and prescription content.")
        # 尝试打印一些样本来帮助调试
        for i, item in enumerate(all_train_data[:5]):
            logger.info(f"Sample {i} data: {item}")
        raise ValueError("Training data preparation resulted in 0 samples")
    
    logger.info(f"Final training samples count: {len(training_data)}")
    return training_data, herbs_list, val_data

def preprocess_validation_data(config, herbs_list, val_data):
    """处理验证数据"""
    # 尝试加载任务1预测结果（如果有）
    task1_predictions = None
    if os.path.exists(config.task1_predictions):
        logger.info(f"Loading task1 predictions for validation data")
        task1_predictions = load_task1_predictions(config.task1_predictions)
    
    # 准备验证数据
    val_output_file = os.path.join(os.path.dirname(config.output_dir), "val_data.jsonl")
    logger.info(f"Preparing validation data and saving to {val_output_file}")
    validation_data = prepare_evaluation_data(val_data, herbs_list, val_output_file, task1_predictions)
    
    return validation_data

def process_data_for_model(data, tokenizer, max_length):
    """将数据处理为模型所需格式"""
    
    def tokenize_function(examples):
        try:
            # 确保输入文本格式正确
            inputs = examples["input"] if isinstance(examples["input"], list) else [examples["input"]]
            
            # 初始化返回结果
            model_inputs = {
                "input_ids": [],
                "attention_mask": []
            }
            
            # 如果有输出文本，合并输入和输出为单个序列（用于因果语言模型）
            if "output" in examples:
                outputs = examples["output"] if isinstance(examples["output"], list) else [examples["output"]]
                
                # 对于因果语言模型，我们需要将输入和输出连接在一起
                combined_texts = []
                for i in range(len(inputs)):
                    # 确保索引在范围内
                    output_text = outputs[i] if i < len(outputs) else ""
                    combined_texts.append(inputs[i] + output_text)
                
                # 编码组合后的文本
                tokenized_inputs = tokenizer(
                    combined_texts,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True,
                    padding=False,  # 不在这里填充，在collator中统一处理
                    return_tensors=None
                )
                
                model_inputs = tokenized_inputs
                
                # 为因果语言模型设置标签，与输入相同
                model_inputs["labels"] = model_inputs["input_ids"].copy()
                
            else:
                # 仅有输入的情况
                tokenized_inputs = tokenizer(
                    inputs,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None
                )
                
                model_inputs = tokenized_inputs
            
            return model_inputs
            
        except Exception as e:
            logger.error(f"Error in tokenization: {e}")
            logger.error(f"Input examples: {examples}")
            raise e
    
    # 转换为HuggingFace Dataset
    if isinstance(data, list):
        logger.info(f"Converting {len(data)} samples to Dataset")
        dataset = Dataset.from_list(data)
    else:
        dataset = data
    
    # 应用处理函数
    logger.info(f"Processing dataset with {len(dataset)} samples")
    
    processed_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=8,  # 减小批处理大小，提高稳定性
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    logger.info(f"Dataset processed, final size: {len(processed_dataset)}")
    return processed_dataset

def train_model(config):
    """训练中药处方推荐模型"""
    # 设置随机种子
    set_random_seed(config.seed)
    
    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.submission_file), exist_ok=True)
    
    # 处理训练和验证数据
    training_data, herbs_list, val_data = preprocess_training_data(config)
    validation_data = preprocess_validation_data(config, herbs_list, val_data)
    
    # 加载分词器和模型
    tokenizer, model = load_tokenizer_and_model(config)
    
    # 准备模型所需的数据格式
    logger.info("Processing data for model training")
    train_dataset = process_data_for_model(training_data, tokenizer, config.max_length)
    
    # 如果有验证集，也进行处理
    val_dataset = None
    if validation_data:
        val_data_with_refs = []
        for item in validation_data:
            if "reference" in item:
                val_data_with_refs.append({
                    "input": item["input"],
                    "output": item["reference"]
                })
        
        if val_data_with_refs:
            val_dataset = process_data_for_model(val_data_with_refs, tokenizer, config.max_length)
    
    # 定义自定义数据整理器函数 - 这将替代之前使用的DataCollatorForSeq2Seq
    def custom_data_collator(features):
        batch = {}
        
        # 处理输入
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        
        # 动态填充到最长序列
        max_length = max(len(ids) for ids in input_ids)
        
        # 右侧填充输入
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * padding_length)
            padded_attention_mask.append(mask + [0] * padding_length)
        
        batch["input_ids"] = torch.tensor(padded_input_ids)
        batch["attention_mask"] = torch.tensor(padded_attention_mask)
        
        # 处理标签（对于CLM模型，标签与输入相同但需要将padding替换为-100）
        if "labels" in features[0]:
            labels = [feature["labels"] for feature in features]
            padded_labels = []
            
            for label, mask in zip(labels, padded_attention_mask):
                # 填充标签到相同长度
                padding_length = max_length - len(label)
                padded_label = label + [-100] * padding_length
                
                # 将padding位置设为-100
                for i in range(len(padded_label)):
                    if i >= len(label) or mask[i] == 0:
                        padded_label[i] = -100
                
                padded_labels.append(padded_label)
            
            batch["labels"] = torch.tensor(padded_labels)
        
        return batch
    
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=10,
        save_strategy="steps",
        save_steps=config.eval_steps * 2,  # 确保save_steps是eval_steps的倍数
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=config.eval_steps if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        save_total_limit=3,
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )
    
    # 定义训练器，使用自定义数据整理器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
    )
    
    # 开始训练
    logger.info("Starting model training")
    trainer.train()
    
    # 保存模型
    logger.info(f"Saving model to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # 保存herbs_list以供推理使用
    with open(os.path.join(config.output_dir, "herbs_list.json"), "w", encoding="utf-8") as f:
        json.dump(herbs_list, f, ensure_ascii=False, indent=2)
    
    logger.info("Training completed")
    return model, tokenizer, herbs_list

if __name__ == "__main__":
    config = Config()
    train_model(config) 