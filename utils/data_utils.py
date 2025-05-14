import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset
import os
import sys
import logging
import random

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.config import Config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class TCMDataset(Dataset):
    """自定义中医数据集"""
    
    def __init__(self, data_file, tokenizer, config=None, is_test=False):
        self.config = Config() if config is None else config
        self.data = self.load_data(data_file)
        self.tokenizer = tokenizer
        self.syndrome_label_map = {label: idx for idx, label in enumerate(self.config.SYNDROME_LABELS)}
        self.disease_label_map = {label: idx for idx, label in enumerate(self.config.DISEASE_LABELS)}
        
        # 是否为测试数据集
        self.is_test = is_test
        if not is_test:
            # 检查是否有标签（只对非测试集检查）
            self.has_labels = self.data and '证型' in self.data[0] and '疾病' in self.data[0]
        else:
            # 测试集假定没有标签
            self.has_labels = False
            logger.info(f"Dataset from {data_file} is marked as test dataset. No labels expected.")
        
    def load_data(self, data_file):
        """加载JSON数据"""
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 整合更丰富的患者信息
        text = self.prepare_text(item)
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.config.MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'id': item['ID'],
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        # 如果有标签数据，则添加到结果中
        if self.has_labels:
            try:
                # 处理多个证型标签
                syndrome_labels = torch.zeros(self.config.NUM_SYNDROME_LABELS)
                
                # 检查证型字段是否存在且非空
                if '证型' in item and item['证型']:
                    syndromes = item['证型'].split('|')
                    # 过滤掉空字符串并处理无效的证型
                    for syndrome in syndromes:
                        if syndrome and syndrome in self.syndrome_label_map:
                            syndrome_idx = self.syndrome_label_map[syndrome]
                            syndrome_labels[syndrome_idx] = 1
                        elif syndrome:
                            logger.warning(f"Unknown syndrome: '{syndrome}' in item ID {item['ID']}")
                else:
                    logger.warning(f"Empty or missing syndrome field in item ID {item['ID']}")
                    # 如果没有证型，使用第一个证型作为默认值
                    syndrome_labels[0] = 1
                
                # 处理疾病标签 (单标签)
                disease_labels = torch.zeros(self.config.NUM_DISEASE_LABELS)
                
                # 检查疾病字段是否存在且有效
                if '疾病' in item and item['疾病'] in self.disease_label_map:
                    disease_idx = self.disease_label_map[item['疾病']]
                    disease_labels[disease_idx] = 1
                else:
                    logger.warning(f"Empty, missing or invalid disease field in item ID {item['ID']}")
                    # 如果没有疾病或疾病无效，使用第一个疾病作为默认值
                    disease_labels[0] = 1
                
                result.update({
                    'syndrome_labels': syndrome_labels,
                    'disease_labels': disease_labels,
                    'raw_syndromes': item.get('证型', ''),
                    'raw_disease': item.get('疾病', '')
                })
            except Exception as e:
                logger.error(f"Error processing item ID {item['ID']}: {str(e)}")
                # 出错时使用默认标签
                syndrome_labels = torch.zeros(self.config.NUM_SYNDROME_LABELS)
                syndrome_labels[0] = 1  # 默认第一个证型
                
                disease_labels = torch.zeros(self.config.NUM_DISEASE_LABELS)
                disease_labels[0] = 1  # 默认第一个疾病
                
                result.update({
                    'syndrome_labels': syndrome_labels,
                    'disease_labels': disease_labels,
                    'raw_syndromes': '',
                    'raw_disease': ''
                })
        elif self.is_test:
            # 对于测试数据，添加默认的标签占位符，避免后续处理出错
            syndrome_labels = torch.zeros(self.config.NUM_SYNDROME_LABELS)
            syndrome_labels[0] = 1  # 默认第一个证型
            
            disease_labels = torch.zeros(self.config.NUM_DISEASE_LABELS)
            disease_labels[0] = 1  # 默认第一个疾病
            
            result.update({
                'syndrome_labels': syndrome_labels,
                'disease_labels': disease_labels,
                'raw_syndromes': '',
                'raw_disease': ''
            })
        
        return result
    
    def prepare_text(self, item):
        """整合患者信息"""
        try:
            text = f"患者信息：{item.get('性别', '')} {item.get('年龄', '')} {item.get('职业', '')} "
            text += f"主诉：{item.get('主诉', '')} "
            text += f"症状：{item.get('症状', '')} "
            text += f"中医望闻切诊：{item.get('中医望闻切诊', '')} "
            
            # 限制病史长度，避免文本过长
            history = item.get('病史', '')[:500] if item.get('病史') else ''
            text += f"病史：{history} "
            
            # 添加辅助检查内容
            aux_exam = item.get('辅助检查', '')[:200] if item.get('辅助检查') else ''
            if aux_exam:
                text += f"辅助检查：{aux_exam}"
                
            return text
        except Exception as e:
            logger.error(f"Error preparing text for item ID {item.get('ID', 'unknown')}: {str(e)}")
            # 返回基本信息，确保不会因为某些特殊数据导致处理失败
            return f"症状：{item.get('症状', '')} 中医望闻切诊：{item.get('中医望闻切诊', '')}"

def split_train_val_dataset(dataset, val_ratio=0.2, seed=42):
    """
    将数据集划分为训练集和验证集
    
    Args:
        dataset: 要划分的数据集
        val_ratio: 验证集比例，默认0.2
        seed: 随机种子，确保可重复性
        
    Returns:
        train_dataset, val_dataset: 划分后的训练集和验证集
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 获取数据集大小
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    
    # 创建随机索引
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    # 划分训练集和验证集
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # 创建数据子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    return train_dataset, val_dataset

def get_dataloaders(tokenizer, config=None, val_ratio=0.2):
    """创建训练和验证数据加载器"""
    config = Config() if config is None else config
    
    # 创建完整训练数据集
    full_train_dataset = TCMDataset(config.TRAIN_FILE, tokenizer, config)
    
    # 划分训练集和验证集
    train_dataset, val_dataset = split_train_val_dataset(full_train_dataset, val_ratio=val_ratio, seed=config.SEED)
    
    # 计算证型类权重
    syndrome_weights = None
    if config.USE_CLASS_WEIGHTS:
        syndrome_weights = calculate_class_weights(full_train_dataset, config.NUM_SYNDROME_LABELS)
    
    # 创建训练数据加载器
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # 创建验证数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return train_dataloader, val_dataloader, None, syndrome_weights

def load_test_dataloader(tokenizer, config=None):
    """加载测试数据集"""
    config = Config() if config is None else config
    
    # 创建测试数据集
    test_dataset = TCMDataset(config.DEV_FILE, tokenizer, config, is_test=True)
    
    # 创建测试数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return test_dataloader

def calculate_class_weights(dataset, num_classes):
    """计算类权重以处理不平衡问题"""
    logger.info("Calculating class weights for imbalanced dataset")
    class_counts = torch.zeros(num_classes)
    
    for i in range(len(dataset)):
        data_item = dataset[i]
        if 'syndrome_labels' in data_item:
            labels = data_item['syndrome_labels']
            class_counts += labels
    
    # 防止除以零
    class_counts = torch.clamp(class_counts, min=1)
    
    # 计算权重 (反比于频率)
    total_samples = len(dataset)
    class_weights = total_samples / (num_classes * class_counts)
    
    # 归一化权重
    class_weights = class_weights / class_weights.sum() * num_classes
    
    logger.info(f"Class weights: {class_weights}")
    return class_weights 