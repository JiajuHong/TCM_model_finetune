import torch
import os

class Config:
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # 预训练模型配置
    # 可选模型: 'bert-base-chinese', 'hfl/chinese-roberta-wwm-ext', 'SJTU-CML/huatuo-llama2-med-7b'
    PRETRAINED_MODEL = 'hfl/chinese-roberta-wwm-ext'
    CACHE_DIR = os.path.join(MODELS_DIR, 'pretrained')
    
    # 数据集配置
    TRAIN_FILE = os.path.join(DATASET_DIR, 'TCM-TBOSD-train.json')
    DEV_FILE = os.path.join(DATASET_DIR, 'TCM-TBOSD-test-B.json')
    SUBMISSION_EXAMPLE_FILE = os.path.join(DATASET_DIR, 'TCM-TBOSD-A.json')
    
    # 训练配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_SEQ_LENGTH = 256  # 增加序列长度
    TRAIN_BATCH_SIZE = 8  # 增加批处理大小
    EVAL_BATCH_SIZE = 8
    LEARNING_RATE = 2e-5  # 调整学习率
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.1
    SEED = 42
    
    # 模型保存配置
    OUTPUT_DIR = os.path.join(MODELS_DIR, 'finetuned')
    SAVE_STEPS = 100
    LOGGING_STEPS = 50
    
    # 标签配置
    SYNDROME_LABELS = [
        '气虚血瘀证', '痰瘀互结证', '气阴两虚证', '气滞血瘀证', 
        '肝阳上亢证', '阴虚阳亢证', '痰热蕴结证', '痰湿痹阻证', 
        '阳虚水停证', '肝肾阴虚证'
    ]
    
    DISEASE_LABELS = [
        '胸痹心痛病', '心衰病', '眩晕病', '心悸病'
    ]
    
    NUM_SYNDROME_LABELS = len(SYNDROME_LABELS)
    NUM_DISEASE_LABELS = len(DISEASE_LABELS)
    
    # 特殊参数
    MIN_SYNDROME_LABELS = 1  # 每个样本至少有多少个证型
    MAX_SYNDROME_LABELS = 2  # 每个样本最多有多少个证型
    
    # 用于动态阈值的配置
    THRESHOLD_RATIO = 0.2  # 差值阈值比例
    
    # 是否使用证型-疾病联合建模
    USE_JOINT_MODELING = True
    
    # 是否使用类权重
    USE_CLASS_WEIGHTS = True 