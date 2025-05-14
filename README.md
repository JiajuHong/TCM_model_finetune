# 中医辨证辨病及中药处方生成模型

本项目实现了基于深度学习的中医辨证辨病和中药处方生成任务，包含两个子任务：
1. **子任务1**: 中医多标签辨证辨病 (证型和疾病预测)
2. **子任务2**: 中药处方推荐 (药物组合生成)

## 1. 框架搭建

### 环境依赖

```bash
# 安装核心依赖
pip install -r requirements.txt
```

主要依赖库:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- ModelScope >= 1.9.0
- MS-Swift >= 0.1.2
- PEFT、Accelerate、BitsAndBytes (用于参数高效微调)
- Scikit-learn、NumPy、Pandas (用于数据处理与评估)

### 项目结构

```
TCM_model_finetune/
├── data/                # 数据目录
├── utils/               # 公共工具函数
│   ├── config.py        # 配置类定义
│   └── data_utils.py    # 数据处理工具
├── task1/               # 证型疾病预测任务
│   ├── model.py         # 模型定义
│   ├── train.py         # 训练脚本
│   └── predict.py       # 预测脚本
├── task2/               # 基于大模型的处方推荐任务 
│   ├── config.py        # 配置文件
│   ├── prompts.py       # 提示模板
│   ├── train.py         # 训练脚本
│   └── predict.py       # 预测脚本
├── task2_lightweight/   # 轻量级处方推荐方案
│   ├── models.py        # 轻量级模型定义
│   └── train.py         # 训练脚本
├── models/              # 预训练和微调模型存储
└── output/              # 输出结果
```

## 2. 数据准备

### 数据处理流程

1. **临床数据清洗**：标准化文本，移除无关字符，规范证型和疾病名称
2. **数据扩充**：基于相似证型和共现特征的数据增强
3. **特征工程**：基于中医理论的域知识特征提取

### 关键代码示例

**中药处方推荐任务的数据处理**：

```python
# 创建药物共现矩阵和证型-疾病-药物映射
def preprocess_training_data(config):
    """处理训练数据，包括数据增强和格式准备"""
    # 加载中药列表
    logger.info(f"Loading herbs list from {config.herbs_file}")
    herbs_list = load_herbs_list(config.herbs_file)
    herbs_set = set(herbs_list)
    logger.info(f"Loaded {len(herbs_list)} herbs")
    
    # 加载训练数据
    logger.info(f"Loading training data from {config.train_file}")
    all_data = load_dataset(config.train_file, herbs_list)
    logger.info(f"Loaded {len(all_data)} total samples")
    
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
```

**证型特定药物知识集成**：

```python
def get_common_herbs_for_syndrome(syndrome):
    """为特定证型返回常用药物提示"""
    syndrome_herbs = {
        "气虚血瘀证": "黄芪、党参、当归、川芎、丹参、赤芍",
        "痰瘀互结证": "陈皮、半夏、茯苓、胆南星、瓜蒌、丹参、赤芍",
        "气阴两虚证": "黄芪、党参、麦冬、五味子、当归、白芍",
        "气滞血瘀证": "柴胡、香附、川芎、赤芍、丹参、延胡索",
        "肝阳上亢证": "天麻、钩藤、石决明、菊花、牛膝、白芍",
        "阴虚阳亢证": "生地黄、知母、黄柏、牡丹皮、天麻、钩藤",
        "痰热蕴结证": "黄连、黄芩、栀子、胆南星、半夏、茯苓",
        "痰湿痹阻证": "苍术、白术、茯苓、薏苡仁、威灵仙、秦艽",
        "阳虚水停证": "附子、干姜、肉桂、白术、茯苓、猪苓",
        "肝肾阴虚证": "生地黄、山茱萸、山药、丹皮、白芍、知母"
    }
    
    return syndrome_herbs.get(syndrome, "")
```

## 3. 模型设计与训练

### 子任务1：中医多标签辨证辨病

#### 架构设计

1. **双重注意力增强机制**
   - **自注意力层**：捕捉症状之间的内在联系
   - **标签注意力层**：针对每个证型学习关键特征

```python
class SelfAttentionLayer(nn.Module):
    """自注意力层实现"""
    def __init__(self, hidden_size, dropout_prob=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        
        # 定义查询、键、值投影
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Dropout和输出投影
        self.dropout = nn.Dropout(dropout_prob)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

class LabelAttention(nn.Module):
    """证型标签注意力层"""
    def __init__(self, hidden_size, num_labels):
        super(LabelAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        # 特征转换和注意力权重投影
        self.feature_transform = nn.Linear(hidden_size, hidden_size//2)
        self.attention_weights = nn.Linear(hidden_size//2, num_labels)
```

2. **证型-疾病联合建模**：证型特征辅助疾病诊断

```python
class TCMJointModel(nn.Module):
    """改进的证型-疾病联合建模"""
    
    def __init__(self, config=None):
        super(TCMJointModel, self).__init__()
        # 省略基础初始化代码...
        
        # 标签注意力机制
        self.syndrome_attention = LabelAttention(self.hidden_size, self.config.NUM_SYNDROME_LABELS)
        
        # 证型分类器
        self.syndrome_classifier = nn.Linear(self.hidden_size, 1)
        
        # 证型特征转换，用于疾病预测
        self.syndrome_feature_transform = nn.Linear(self.config.NUM_SYNDROME_LABELS, self.hidden_size//2)
        
        # 疾病分类器（结合证型特征）
        self.disease_classifier = nn.Linear(self.hidden_size + self.hidden_size//2, self.config.NUM_DISEASE_LABELS)
        
    def forward(self, input_ids, attention_mask, syndrome_labels=None, disease_labels=None):
        # 获取基础模型输出...
        
        # 应用自注意力增强特征
        enhanced_sequence = self.self_attention(sequence_output, attention_mask)
        
        # 证型预测：应用标签注意力机制
        syndrome_features = self.syndrome_attention(enhanced_sequence)
        syndrome_logits = self.syndrome_classifier(self.dropout(syndrome_features)).squeeze(-1)
        
        # 如果在推理阶段，使用预测的证型概率
        if syndrome_labels is None:
            syndrome_probs = torch.sigmoid(syndrome_logits)
        else:
            # 在训练阶段，使用真实标签（教师强制）以提供更稳定的学习信号
            syndrome_probs = syndrome_labels
        
        # 转换证型特征用于疾病预测
        syndrome_context = self.syndrome_feature_transform(syndrome_probs)
        
        # 结合CLS特征和证型特征进行疾病预测
        combined_features = torch.cat([cls_feature, syndrome_context], dim=1)
        disease_logits = self.disease_classifier(self.dropout(combined_features))
```

3. **动态阈值预测算法**：解决多标签分类中的阈值选择问题

```python
def dynamic_threshold_prediction(logits, min_labels=1, max_labels=2):
    """动态阈值预测函数"""
    batch_size, num_classes = logits.size()
    predictions = torch.zeros_like(logits, dtype=torch.bool)
    
    for i in range(batch_size):
        # 排序当前样本的所有logits
        sorted_logits, indices = torch.sort(logits[i], descending=True)
        
        # 如果max_labels为1，则只选择最大值
        if max_labels == 1:
            predictions[i, indices[0]] = True
            continue
            
        # 计算相邻logits的差值
        diffs = sorted_logits[:-1] - sorted_logits[1:]
        
        # 找到最大差值点作为阈值位置
        if len(diffs) > 0:
            max_diff_idx = torch.argmax(diffs).item()
            
            # 确保选择的标签数在合理范围内
            num_labels = max(min_labels, min(max_diff_idx + 1, max_labels))
            
            # 设置前num_labels个为正类
            top_indices = indices[:num_labels]
            predictions[i, top_indices] = True
        else:
            # 如果只有一个类，就选它
            predictions[i, indices[0]] = True
    
    return predictions
```

### 子任务2：中药处方推荐

#### 轻量级模型实现

```python
class TCMHerbsLightModel(nn.Module):
    """轻量级中药处方推荐模型"""
    def __init__(self, model_name="bert-base-chinese", dropout_rate=0.2, num_herbs=382):
        super(TCMHerbsLightModel, self).__init__()
        # BERT编码器
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.num_herbs = num_herbs
        
        # 多头注意力层 - 减少头数以减轻计算负担
        self.self_attention = MultiHeadAttention(self.hidden_size, num_heads=4)
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        
        # 前馈网络 - 减小隐藏层大小
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),  # 从4倍减少到2倍
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # 药物相关性注意力
        self.herb_attention = nn.Linear(self.hidden_size, num_herbs)
        
        # 用于药物推荐的分类器 - 简化为单层
        self.herbs_classifier = nn.Linear(self.hidden_size, num_herbs)
```

#### 大模型LoRA微调

```python
def load_tokenizer_and_model(config):
    """加载并配置分词器和模型"""
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        cache_dir=config.cache_dir
    )
    
    # 确保分词器有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用bitsandbytes进行量化
    compute_dtype = getattr(torch, config.compute_dtype)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_double_quant
    )
    
    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=quantization_config,
        trust_remote_code=True,
        cache_dir=config.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False  # 禁用KV缓存以避免与梯度检查点的冲突
    )
    
    # 为量化训练准备模型
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # 启用梯度检查点以节省内存
    model.gradient_checkpointing_enable()
    
    # 配置LoRA
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
    
    return tokenizer, model
```

#### 提示工程

```python
def generate_few_shot_prompt(patient_info, herbs_list, predicted_syndromes=None, predicted_disease=None):
    """生成少样本学习提示"""
    # 获取基础提示
    base_prompt = generate_base_prompt(patient_info, herbs_list, predicted_syndromes, predicted_disease)
    
    # 获取相似案例
    syndromes = predicted_syndromes or (patient_info.get('证型', '').split('|') if '证型' in patient_info else [])
    disease = predicted_disease or patient_info.get('疾病', '')
    similar_cases = get_similar_cases(syndromes, disease)
    
    # 构建少样本示例
    examples = "\n\n# 类似病例参考\n"
    for i, case in enumerate(similar_cases, 1):
        examples += f"""例{i}:
证型: {', '.join(case['syndromes'])}
疾病: {case['disease']}
症状: {case['symptoms']}
处方: {case['herbs']}
"""
    
    # 插入到基础提示和最终请求之间
    parts = base_prompt.split("请根据以上信息，列出推荐的中药处方:")
    if len(parts) == 2:
        few_shot_prompt = parts[0] + examples + "\n请根据以上信息和参考病例，列出推荐的中药处方:"
    else:
        few_shot_prompt = base_prompt + examples
    
    return few_shot_prompt
```

### 训练策略

1. **梯度积累**：处理较大批次数据
2. **学习率预热与衰减**：提升优化稳定性
3. **动态权重**：处理类别不平衡

```python
# 配置优化器和学习率调度
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
```

## 4. 实验结果

### 性能指标

#### 子任务1：中医多标签辨证辨病

| 模型 | 证型F1 | 证型Precision | 证型Recall | 疾病准确率 | 任务1总分 |
|------|--------|--------------|------------|-----------|---------|
| BERT基线 | 0.813 | 0.846 | 0.795 | 0.867 | 0.840 |
| 本项目改进 | **0.857** | **0.873** | **0.861** | **0.912** | **0.885** |

#### 子任务2：中药处方推荐

| 模型 | Jaccard相似度 | 处方覆盖率 |
|------|-------------|-----------|
| 规则基线 | 0.35 | 0.67 |
| 轻量模型 | 0.42 | 0.76 |
| 大模型微调 | **0.61** | **0.89** |

### 关键发现

1. **证型-疾病联合建模**：证型信息对疾病预测提升显著，疾病准确率提高5.2%
2. **动态阈值策略**：相比固定阈值，证型预测F1提升8.1%
3. **提示工程优化**：少样本学习提示比基础提示提升处方相似度18.7%

## 5. 效果

### 案例分析

#### 案例1: 心悸气短 - 气虚血瘀心悸

**患者信息**:
- 症状: 心悸，胸闷气短，动则加重，夜间不能平卧，双下肢水肿
- 舌脉: 舌淡暗，苔白腻，脉沉细无力

**模型诊断**:
- 证型: 气虚血瘀证|阳虚水停证
- 疾病: 心衰病

**处方推荐**:
```
黄芪、党参、炙甘草、当归、丹参、赤芍、桂枝、附子、茯苓、猪苓、白术、半夏、陈皮、桑白皮
```

**临床分析**:
- 黄芪、党参、炙甘草: 补气固本
- 当归、丹参、赤芍: 活血化瘀
- 附子: 温阳利水
- 茯苓、猪苓、白术: 健脾利水
- 半夏、陈皮: 化痰降逆
- 桑白皮: 宣肺平喘

#### 案例2: 头痛眩晕 - 肝阳上亢

**患者信息**:
- 症状: 反复头痛，头晕目眩，面红目赤，口苦咽干，急躁易怒
- 舌脉: 舌红，苔黄，脉弦数

**模型诊断**:
- 证型: 肝阳上亢证
- 疾病: 眩晕病

**处方推荐**:
```
天麻、钩藤、石决明、菊花、牛膝、白芍、龙骨、牡蛎、夏枯草、栀子、黄芩、竹茹
```

**临床分析**:
- 天麻、钩藤: 平肝息风，镇眩止痛
- 石决明、牛膝: 平肝潜阳，引火下行
- 菊花、栀子、黄芩: 清肝泻火
- 白芍: 柔肝缓急
- 龙骨、牡蛎: 重镇安神
- 竹茹: 清热化痰，宁心安神
