import json
import os
import re
import random
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader

# 全局变量定义
SYNDROME_CLASSES = ['气虚血瘀证', '痰瘀互结证', '气阴两虚证', '气滞血瘀证', '肝阳上亢证', 
                 '阴虚阳亢证', '痰热蕴结证', '痰湿痹阻证', '阳虚水停证', '肝肾阴虚证']
DISEASE_CLASSES = ['胸痹心痛病', '心衰病', '眩晕病', '心悸病']

def clean_text(text):
    """Clean text"""
    # Remove special characters
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def text_augmentation(text):
    """Text augmentation"""
    # Randomly delete some characters
    if random.random() < 0.3:
        words = list(text)
        n_delete = int(len(words) * 0.1)
        for _ in range(n_delete):
            if words:
                del words[random.randint(0, len(words)-1)]
        text = ''.join(words)
    return text

# 加载中药列表
def load_herbs():
    with open('all_herbs.txt', 'r', encoding='utf-8') as f:
        herbs = [line.strip() for line in f.readlines()]
    return herbs

# 合并病例文本信息
def merge_case_text(case_data):
    """合并一个病例的所有文本字段为一个字符串"""
    text_fields = [
        '性别', '职业', '年龄', '婚姻', '病史陈述者', '发病节气',
        '主诉', '症状', '中医望闻切诊', '病史', '体格检查', '辅助检查'
    ]
    
    text = ""
    for field in text_fields:
        if field in case_data and case_data[field]:
            text += f"{field}: {case_data[field]} "
    
    return text.strip()

# 数据集类
class TCMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, task_type="syndrome_disease", herbs_list=None, is_training=True):
        """
        初始化数据集
        
        参数:
        data_path: 数据文件路径
        tokenizer: tokenizer对象
        max_length: 最大序列长度
        task_type: 任务类型，"syndrome_disease"或"herbs"或"joint"
        herbs_list: 中药列表，仅当task_type="herbs"或"joint"时需要
        is_training: 是否为训练模式
        """
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.is_training = is_training
        
        # 读取数据
        with open(data_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        
        # 重要字段权重
        field_weights = {
            '主诉': 2,
            '症状': 2,
            '中医望闻切诊': 2,
            '病史': 1.5,
            '体格检查': 1.5,
            '辅助检查': 1
        }
        
        # 为处方任务准备标签编码器
        if (task_type == "herbs" or task_type == "joint") and herbs_list:
            self.herbs_list = herbs_list
            self.mlb = MultiLabelBinarizer(classes=herbs_list)
            self.mlb.fit([herbs_list])  # 拟合所有可能的中药
        
        for case in cases:
            # 构建加权文本
            text = ""
            for field, weight in field_weights.items():
                if field in case and case[field]:
                    # 重复添加重要字段
                    text += (case[field] + " ") * int(weight)
            
            # 添加其他字段
            for field in ['性别', '职业', '年龄', '婚姻', '病史陈述者', '发病节气']:
                if field in case and case[field]:
                    text += f"{field}: {case[field]} "
            
            # 清理文本
            text = clean_text(text)
            
            # 数据增强
            if self.is_training:
                text = text_augmentation(text)

            # 基本项目: ID和文本
            item = {
                'ID': case['ID'],
                'text': text
            }
            
            # 添加证型/疾病标签（针对syndrome_disease或joint任务）
            if task_type in ["syndrome_disease", "joint"] and '证型' in case and '疾病' in case and case['证型'] and case['疾病']:
                # 分割可能的多个证型和疾病
                syndromes = case['证型'].split('|') if isinstance(case['证型'], str) else case['证型']
                diseases = [case['疾病']] if isinstance(case['疾病'], str) else case['疾病']
                
                item['syndromes'] = syndromes
                item['diseases'] = diseases
            
            # 添加处方标签（针对herbs或joint任务）
            if task_type in ["herbs", "joint"] and '处方' in case and case['处方']:
                # 处理处方
                herbs = case['处方']
                if isinstance(herbs, str):
                    try:
                        herbs = eval(herbs)  # 尝试解析字符串列表
                    except:
                        herbs = [herbs]
                
                item['herbs'] = herbs
            
            # 根据任务类型添加到数据集
            if (task_type == "syndrome_disease" and 'syndromes' in item and 'diseases' in item) or \
               (task_type == "herbs" and 'herbs' in item) or \
               (task_type == "joint" and 'syndromes' in item and 'diseases' in item and 'herbs' in item):
                self.data.append(item)
    
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
        inputs = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ID': item['ID']
        }
        
        # 根据任务类型处理标签
        if self.task_type in ["syndrome_disease", "joint"]:
            # 证型标签 one-hot 编码
            syndrome_labels = torch.zeros(len(SYNDROME_CLASSES))
            for syndrome in item['syndromes']:
                if syndrome in SYNDROME_CLASSES:
                    syndrome_labels[SYNDROME_CLASSES.index(syndrome)] = 1
            
            # 疾病标签 one-hot 编码
            disease_labels = torch.zeros(len(DISEASE_CLASSES))
            for disease in item['diseases']:
                if disease in DISEASE_CLASSES:
                    disease_labels[DISEASE_CLASSES.index(disease)] = 1
            
            inputs['syndrome_labels'] = syndrome_labels
            inputs['disease_labels'] = disease_labels
            
        if self.task_type in ["herbs", "joint"]:
            # 中药标签 one-hot 编码
            herbs_labels = self.mlb.transform([item['herbs']])[0]
            inputs['herbs_labels'] = torch.FloatTensor(herbs_labels)
        
        return inputs

# 加载数据集
def load_data(train_path, test_path, tokenizer, batch_size=8, max_length=512, task_type="syndrome_disease", herbs_list=None):
    """
    加载训练集和测试集
    
    参数:
    train_path: 训练集路径
    test_path: 测试集路径
    tokenizer: tokenizer对象
    batch_size: 批次大小
    max_length: 最大序列长度
    task_type: 任务类型，"syndrome_disease"或"herbs"或"joint"
    herbs_list: 中药列表
    """
    # 根据任务类型确定是否需要herbs_list
    if task_type in ["herbs", "joint"] and herbs_list is None:
        herbs_list = load_herbs()
        print(f"自动加载中药列表: {len(herbs_list)}种中药")
    
    train_dataset = TCMDataset(train_path, tokenizer, max_length, task_type, herbs_list, is_training=True)
    test_dataset = None
    
    if test_path and os.path.exists(test_path):
        test_dataset = TCMDataset(test_path, tokenizer, max_length, task_type, herbs_list, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Load test data
def load_test_data(test_path, tokenizer, max_length=512):
    """
    Load test data for prediction
    
    Args:
        test_path: Path to test file
        tokenizer: Tokenizer object
        max_length: Maximum sequence length
    """
    try:
        test_data = []
        
        with open(test_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        
        # Important field weights
        field_weights = {
            '主诉': 2,
            '症状': 2,
            '中医望闻切诊': 2,
            '病史': 1.5,
            '体格检查': 1.5,
            '辅助检查': 1
        }
        
        for case in cases:
            # Build weighted text
            text = ""
            for field, weight in field_weights.items():
                if field in case and case[field]:
                    # Repeat important fields
                    text += (case[field] + " ") * int(weight)
            
            # Add other fields
            for field in ['性别', '职业', '年龄', '婚姻', '病史陈述者', '发病节气']:
                if field in case and case[field]:
                    text += f"{field}: {case[field]} "
            
            # Clean text
            text = clean_text(text)
            
            test_data.append({
                'ID': case['ID'],
                'text': text
            })
            
        return test_data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

# Generate submission format
def generate_submission(predictions_task1, predictions_task2, output_file):
    """
    Generate submission file in the required format
    
    Args:
        predictions_task1: Task 1 predictions, format {ID: [syndrome, disease]}
        predictions_task2: Task 2 predictions, format {ID: [herbs list]}
        output_file: Output file path
    """
    try:
        submission = []
        
        # Merge all IDs
        all_ids = set(list(predictions_task1.keys()) + list(predictions_task2.keys()))
        
        for id in all_ids:
            item = {"ID": id}
            
            # Add task 1 predictions - [syndrome, disease]
            if id in predictions_task1:
                item["子任务1"] = predictions_task1[id]
            
            # Add task 2 predictions - list of herbs
            if id in predictions_task2:
                item["子任务2"] = predictions_task2[id]
            
            submission.append(item)
        
        # Save as JSON with proper formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False, indent=4)
        
        print(f"Submission file saved to {output_file}")
        
        # Print an example for verification
        if submission:
            print("\nSubmission format example:")
            example = json.dumps(submission[0], ensure_ascii=False, indent=4)
            print(example)
            
            # Verification checks
            print("\nVerification:")
            if "子任务1" in submission[0] and isinstance(submission[0]["子任务1"], list) and len(submission[0]["子任务1"]) == 2:
                print("✓ Task 1 format is correct: [syndrome, disease]")
            else:
                print("✗ Task 1 format is incorrect!")
                
            if "子任务2" in submission[0] and isinstance(submission[0]["子任务2"], list):
                print("✓ Task 2 format is correct: list of herbs")
            else:
                print("✗ Task 2 format is incorrect!")
    except Exception as e:
        print(f"Error generating submission file: {e}") 