import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("task2_lightweight.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_herbs_list(herbs_file):
    """从文件加载中药列表"""
    herbs_list = []
    try:
        with open(herbs_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 尝试解析为Python列表
            if content.startswith('[') and content.endswith(']'):
                try:
                    # 使用eval解析列表字符串
                    herbs_list = eval(content)
                    logger.info(f"Successfully parsed herbs list with {len(herbs_list)} items as Python list")
                except Exception as e:
                    logger.warning(f"Failed to parse as Python list: {e}, trying line-by-line mode")
                    # 如果解析失败，回退到按行读取模式
                    with open(herbs_file, 'r', encoding='utf-8') as f2:
                        for line in f2:
                            herb = line.strip()
                            if herb:
                                herbs_list.append(herb)
            else:
                # 按行读取模式
                with open(herbs_file, 'r', encoding='utf-8') as f2:
                    for line in f2:
                        herb = line.strip()
                        if herb:
                            herbs_list.append(herb)
                
            logger.info(f"Loaded {len(herbs_list)} herbs from {herbs_file}")
        return herbs_list
    except Exception as e:
        logger.error(f"Error loading herbs list: {e}")
        return []

def clean_text(text):
    """清理文本，移除特殊字符"""
    if not text:
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # 规范化空白字符
    return text

def extract_text_from_case(case_data):
    """从病例数据中提取文本特征"""
    # 重要字段权重
    field_weights = {
        '主诉': 2.0,
        '症状': 2.0,
        '中医望闻切诊': 2.0,
        '病史': 1.5,
        '体格检查': 1.5,
        '辅助检查': 1.0,
        '证型': 2.0,
        '疾病': 2.0
    }
    
    text = ""
    
    # 添加带权重的重要字段
    for field, weight in field_weights.items():
        if field in case_data and case_data[field]:
            content = case_data[field]
            # 对于列表类型字段，进行合并
            if isinstance(content, list):
                content = '，'.join(content)
            # 重复重要字段以增加其权重
            text += f"{field}: {content} " * int(weight)
    
    # 添加基本信息字段
    basic_fields = ['性别', '年龄', '职业', '婚姻', '发病节气']
    for field in basic_fields:
        if field in case_data and case_data[field]:
            text += f"{field}: {case_data[field]} "
    
    # 清理文本
    text = clean_text(text)
    
    return text

class TCMHerbsDataset(Dataset):
    """中药处方推荐数据集"""
    def __init__(self, data_path, tokenizer, max_length=512, herbs_list=None, is_training=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.is_training = is_training
        
        try:
            # 加载数据
            with open(data_path, 'r', encoding='utf-8') as f:
                cases = json.load(f)
            
            logger.info(f"Loaded {len(cases)} cases from {data_path}")
            
            # 收集所有出现的中药名称
            all_herbs = set(herbs_list) if herbs_list else set()
            
            for case in cases:
                item = {
                    'ID': case.get('ID', ''),
                    'text': extract_text_from_case(case)
                }
                
                # 如果是训练数据，需要处理处方标签
                if is_training and '处方' in case:
                    herbs = case['处方']
                    # 处理不同格式的处方
                    if isinstance(herbs, str):
                        try:
                            herbs = eval(herbs)  # 尝试解析字符串列表
                        except:
                            herbs = [herbs]
                    
                    if not isinstance(herbs, list):
                        logger.warning(f"Invalid prescription format for case {item['ID']}: {herbs}")
                        continue
                    
                    # 收集所有中药名称
                    for herb in herbs:
                        all_herbs.add(herb)
                    
                    item['herbs'] = herbs
                    self.data.append(item)
                elif not is_training:
                    # 测试数据不需要处方
                    self.data.append(item)
            
            # 初始化多标签二值化器，使用合并后的中药列表
            self.mlb = MultiLabelBinarizer()
            all_herbs_list = sorted(list(all_herbs))  # 排序以保持一致性
            self.mlb.fit([all_herbs_list])
            logger.info(f"Using {len(all_herbs_list)} unique herbs for classification")
            
            if herbs_list and len(all_herbs) > len(herbs_list):
                new_herbs = all_herbs - set(herbs_list)
                logger.info(f"Added {len(new_herbs)} new herbs from training data: {', '.join(sorted(list(new_herbs)))}")
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.mlb = MultiLabelBinarizer()
            if herbs_list:
                self.mlb.fit([herbs_list])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 使用tokenizer处理文本
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 去除批次维度
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ID': item['ID']
        }
        
        # 如果是训练数据，添加中药标签
        if self.is_training and 'herbs' in item:
            herbs_labels = self.mlb.transform([item['herbs']])[0]
            result['herbs_labels'] = torch.FloatTensor(herbs_labels)
        
        return result

def load_data(train_path, test_path, tokenizer, batch_size=8, max_length=512, herbs_list=None):
    """加载训练集和测试集"""
    try:
        train_dataset = TCMHerbsDataset(
            train_path, 
            tokenizer, 
            max_length=max_length, 
            herbs_list=herbs_list, 
            is_training=True
        )
        
        test_dataset = None
        if test_path and os.path.exists(test_path):
            test_dataset = TCMHerbsDataset(
                test_path, 
                tokenizer, 
                max_length=max_length, 
                herbs_list=list(train_dataset.mlb.classes_),  # 使用训练集中收集的中药列表
                is_training=True  # 如果测试集有标签，设为True
            )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = None
        
        if test_dataset:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 返回从训练数据中提取的完整中药列表
        complete_herbs_list = list(train_dataset.mlb.classes_)
        
        return train_loader, test_loader, train_dataset.mlb, complete_herbs_list
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, []

def evaluate_herbs_prediction(pred_herbs, true_herbs):
    """评估中药处方推荐性能"""
    if not pred_herbs or not true_herbs:
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'jaccard': 0
        }
    
    # 转为集合计算重叠
    pred_set = set(pred_herbs)
    true_set = set(true_herbs)
    
    # 计算交集大小
    intersection = len(pred_set.intersection(true_set))
    
    # 计算精确率、召回率
    precision = intersection / len(pred_set) if pred_set else 0
    recall = intersection / len(true_set) if true_set else 0
    
    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算Jaccard相似度
    jaccard = intersection / len(pred_set.union(true_set)) if pred_set or true_set else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard
    }

def process_herbs_prediction(logits, mlb, threshold=0.5, top_k=15):
    """处理模型输出的药物预测结果"""
    # 根据阈值进行二值化预测
    pred_probs = torch.sigmoid(logits).cpu().numpy()
    
    # 两种选择中药的策略：
    # 1. 按阈值选择
    binary_preds = (pred_probs >= threshold).astype(int)
    
    # 2. 按Top-K选择
    herbs_predictions = []
    for i, probs in enumerate(pred_probs):
        # 策略1: 使用阈值
        threshold_herbs = mlb.inverse_transform(binary_preds[i:i+1])[0]
        
        # 策略2: 使用top-k
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_herbs = [mlb.classes_[idx] for idx in top_indices if probs[idx] > 0.1]  # 添加最小概率阈值
        
        # 如果阈值选择的药物数量合理，则使用阈值策略，否则使用top-k策略
        if 5 <= len(threshold_herbs) <= 20:
            herbs_predictions.append(threshold_herbs)
        else:
            herbs_predictions.append(top_k_herbs)
    
    return herbs_predictions

def load_task1_predictions(file_path):
    """加载任务1的预测结果，用于辅助任务2"""
    try:
        predictions = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            if 'ID' in item and '子任务1' in item:
                predictions[item['ID']] = item['子任务1']
        
        return predictions
    except Exception as e:
        logger.error(f"Error loading task1 predictions: {e}")
        return {} 