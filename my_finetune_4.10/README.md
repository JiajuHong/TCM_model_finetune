
# 中医辨证辨病及中药处方生成

本项目实现了基于临床文本信息的中医辨证辨病和中药处方生成功能，主要包含两个子任务：
1. 中医多标签辨证辨病（预测证型和疾病）
2. 中药处方推荐（推荐中药组合）

## 项目优势

相较于基线模型，本项目具有以下优化：

- **注意力机制**：引入多头注意力和标签感知注意力，提升文本特征提取能力
- **门控融合技术**：使用门控融合机制连接不同特征，增强模型表达能力
- **优化的特征工程**：针对中医文本特点，设计了字段权重策略，突出关键信息
- **动态阈值算法**：自适应阈值选择，提高多标签分类性能
- **灵活的模型选择**：支持单任务或联合任务训练，适应不同场景需求

## 环境配置

### 1. 安装依赖

```bash
pip install torch==2.6.0
pip install transformers==4.18.0
pip install scikit-learn==1.0.2
pip install matplotlib==3.5.1
pip install tqdm==4.64.0
pip install numpy==1.21.5
```

### 2. 数据准备

确保以下文件放置在正确位置：
- `data/TCM-TBOSD-train.json`：训练数据集
- `data/TCM-TBOSD-test-A.json`：测试数据集（A榜验证集）
- `all_herbs.txt`：中药名录（放在项目根目录）

## 项目结构

```
project/
├── data/
│   ├── TCM-TBOSD-train.json  # 训练数据
│   └── TCM-TBOSD-test-A.json # 测试数据
├── models/                   # 训练好的模型存放目录
├── models.py                 # 模型架构定义
├── train.py                  # 训练脚本
├── utils.py                  # 数据处理与辅助函数
├── predict.py                # 预测与生成提交文件
├── all_herbs.txt             # 中药名录（381种）
└── requirements.txt          # 项目依赖
```

## 使用方法

### 1. 训练模型

直接运行train.py文件，无需命令行参数：

```bash
python train.py
```

模型会自动训练证型疾病模型和中药处方模型。如需修改训练参数，请编辑train.py文件中的Args类：

```python
class Args:
    def __init__(self):
        # 数据参数
        self.train_file = 'data/TCM-TBOSD-train.json'
        self.val_file = 'data/TCM-TBOSD-test-A.json'
        self.output_dir = './models'
        
        # 模型参数
        self.model_name = 'hfl/chinese-bert-wwm-ext'  # 预训练模型
        self.max_length = 512                         # 最大序列长度
        self.dropout_rate = 0.2                       # Dropout率
        
        # 训练参数
        self.batch_size = 8                           # 批次大小
        self.epochs = 3                               # 训练轮数
        self.learning_rate = 2e-5                     # 学习率
        self.weight_decay = 0.01                      # 权重衰减
        self.max_grad_norm = 1.0                      # 梯度裁剪阈值
        
        # 任务选择
        self.task = 'both'   # 'syndrome_disease'或'herbs'或'both'
```

可通过修改task参数选择训练特定模型：
- `syndrome_disease`：仅训练证型疾病模型
- `herbs`：仅训练中药处方模型
- `both`：同时训练两个模型（默认）

### 2. 预测并生成提交文件

```bash
python predict.py
```

默认会加载models文件夹中的最佳模型进行预测，生成符合比赛要求的submission.json文件。

## 模型技术细节

### 核心技术

1. **自注意力机制**：
   - 8头多头注意力层，增强对长文本的语义理解能力
   - 残差连接与层归一化，优化梯度流动和训练稳定性

```python
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_size, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 确保维度可以被头数整除
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)
        
        # 投影并重塑为多头格式
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask.float()) * -10000.0
            scores = scores + extended_attention_mask
        
        # Softmax并应用dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算上下文向量
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        # 最终线性投影
        output = self.output_proj(context)
        
        return output
```

2. **标签感知注意力机制(LAAT)**：
   - 针对每个标签学习特定的注意力权重
   - 显著提升多标签分类准确率

```python
# 证型分类中的标签感知注意力实现
# TCMSyndromeDiseaseModel类中的相关实现
def forward(self, input_ids, attention_mask):
    # ... 省略前面代码
    
    # 证型分类 - 使用标签感知注意力机制
    syndrome_features = self.syndrome_feature(hidden_states)
    
    # 计算标签级别的注意力权重
    syndrome_attn_scores = self.syndrome_attn(syndrome_features)  # [batch, seq_len, num_labels]
    
    # 应用掩码并计算权重
    extended_attention_mask = attention_mask.unsqueeze(-1).expand_as(syndrome_attn_scores)
    syndrome_attn_scores = syndrome_attn_scores.masked_fill(extended_attention_mask == 0, -1e9)
    syndrome_attn_weights = F.softmax(syndrome_attn_scores, dim=1)
    
    # 加权求和得到每个标签的表示
    syndrome_weighted_features = torch.bmm(
        syndrome_attn_weights.transpose(1, 2),  # [batch, num_labels, seq_len]
        syndrome_features  # [batch, seq_len, hidden_size]
    )  # [batch, num_labels, hidden_size]
    
    # 生成证型预测
    syndrome_logits = self.syndrome_classifier(self.dropout(syndrome_weighted_features))
    
    # ... 省略后面代码
```

3. **门控融合机制**：
   - 智能融合证型与疾病特征
   - 通过可学习门控参数自适应调整特征权重

```python
class GatedFusion(nn.Module):
    """门控融合机制"""
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x1, x2):
        # 拼接两个输入向量
        concat = torch.cat([x1, x2], dim=-1)
        # 计算门控参数
        gate_value = torch.sigmoid(self.gate(concat))
        # 执行门控融合
        output = gate_value * x1 + (1 - gate_value) * x2
        return output

# 在TCMSyndromeDiseaseModel中的应用
if self.use_fusion:
    disease_context = self.syndrome_disease_fusion(disease_features, syndrome_pooled)
else:
    disease_context = disease_features
```

4. **动态阈值选择**：
   - 使用`find_threshold_micro`函数自动计算最优分类阈值
   - 对不同数据样本动态适应，提高准确率

```python
def find_threshold_micro(logits, labels):
    """寻找最优阈值"""
    logits_1d = logits.reshape(-1)
    labels_1d = labels.reshape(-1)
    sort_arg = np.argsort(logits_1d)
    sort_label = np.take_along_axis(labels_1d, sort_arg, axis=0)
    label_count = np.sum(sort_label)
    correct = label_count - np.cumsum(sort_label)
    predict = labels_1d.shape[0] + 1 - np.cumsum(np.ones_like(sort_label))
    f1 = 2 * correct / (predict + label_count)
    sort_logits = np.take_along_axis(logits_1d, sort_arg, axis=0)
    f1_argmax = np.argmax(f1)
    threshold = sort_logits[f1_argmax]
    return threshold

# 在模型评估中的应用
syndrome_probs = torch.sigmoid(syndrome_logits).cpu().numpy()
syndrome_pred = syndrome_probs > threshold
```

5. **文本特征增强**：
   - 对主诉、症状等关键字段增加权重（2倍）
   - 数据增强技术（随机字符删除）提高模型泛化能力
   - 丰富的文本清理与预处理流程

```python
# 重要字段权重设置
field_weights = {
    '主诉': 2,
    '症状': 2,
    '中医望闻切诊': 2,
    '病史': 1.5,
    '体格检查': 1.5,
    '辅助检查': 1
}

# 构建加权文本
text = ""
for field, weight in field_weights.items():
    if field in case and case[field]:
        # 重复添加重要字段
        text += (case[field] + " ") * int(weight)

# 文本增强函数
def text_augmentation(text):
    """文本增强"""
    # 随机删除部分字符
    if random.random() < 0.3:
        words = list(text)
        n_delete = int(len(words) * 0.1)
        for _ in range(n_delete):
            if words:
                del words[random.randint(0, len(words)-1)]
        text = ''.join(words)
    return text
```

### 模型架构设计

#### 证型疾病模型

- 基础编码器：采用中文BERT-wwm（哈工大版本）
- 特征提取：Transformer层+多头注意力+残差连接
- 证型预测：多标签分类（BCEWithLogitsLoss）
- 疾病预测：单标签分类（CrossEntropyLoss）
- 特点：支持Focal Loss处理类别不平衡问题

```python
# 证型疾病模型核心架构
class TCMSyndromeDiseaseModel(nn.Module):
    def __init__(self, model_name="hfl/chinese-bert-wwm-ext", dropout_rate=0.2, use_fusion=True):
        super(TCMSyndromeDiseaseModel, self).__init__()
        # BERT编码器
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        
        # 多头注意力层
        self.attention = MultiHeadAttention(self.hidden_size, num_heads=8)
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
        
        # 特征增强层
        self.syndrome_feature = nn.Linear(self.hidden_size, self.hidden_size)
        self.disease_feature = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 门控融合
        self.use_fusion = use_fusion
        if use_fusion:
            self.syndrome_disease_fusion = GatedFusion(self.hidden_size)
        
        # 证型分类器 - 使用标签注意力机制
        self.syndrome_attn = nn.Linear(self.hidden_size, len(SYNDROME_CLASSES))
        self.syndrome_classifier = nn.Linear(self.hidden_size, len(SYNDROME_CLASSES))
        
        # 疾病分类器
        self.disease_classifier = nn.Linear(self.hidden_size, len(DISEASE_CLASSES))
```

#### 中药处方模型

- 基础编码器：与证型疾病模型相同
- 药物知识图谱：通过图卷积网络建模药物间关系
- 药物相关性学习：自适应学习药物共现模式
- 多层分类器：处理大规模多标签分类（381种中药）
- 预测优化：灵活控制最小/最大推荐药物数量

```python
# 中药处方模型药物知识图谱实现
def build_herb_graph(self, hidden_states):
    """构建药物知识图谱"""
    # 计算药物之间的相似度
    herb_sim = torch.matmul(self.herbs_emb, self.herbs_emb.transpose(0, 1))
    # 归一化
    herb_sim = F.softmax(herb_sim, dim=-1)
    # 应用图卷积
    herb_features = torch.matmul(herb_sim, torch.matmul(self.herbs_emb, self.gcn_weight))
    return herb_features

# 处方预测关键代码
def predict_herbs(args):
    # ... 省略前面代码
    
    with torch.no_grad():
        for item in tqdm(test_data, desc="Predicting herb prescriptions"):
            # ... 省略前面代码
            
            # 模型预测
            herbs_logits = model(input_ids, attention_mask)
            
            # 获取药物预测结果
            herbs_probs = torch.sigmoid(herbs_logits).cpu().numpy()[0]
            
            # 选择概率最高的N种药物
            top_indices = np.argsort(herbs_probs)[::-1][:args.top_herbs]
            
            # 过滤低于阈值的药物
            selected_indices = [i for i in top_indices if herbs_probs[i] > args.herbs_threshold]
            
            # 如果药物数量不足，补充到最小数量
            if len(selected_indices) < args.min_herbs:
                remaining = args.min_herbs - len(selected_indices)
                for idx in top_indices:
                    if idx not in selected_indices and remaining > 0:
                        selected_indices.append(idx)
                        remaining -= 1
            
            # 转换为药物名称
            predicted_herbs = [herbs_list[i] for i in selected_indices]
```


## 注意事项

1. 初次运行会自动下载预训练模型，请确保网络连接正常
2. 针对大规模训练，建议增加epochs（5-10轮）和调整batch_size
3. 默认使用CUDA加速（如可用），性能最佳
4. 训练过程自动保存最佳模型，可在models目录查看
5. 生成的提交文件完全符合比赛要求格式，可直接提交
6. 代码包含完善的错误处理和日志记录，运行更稳定

## 可能的进一步改进

1. 引入更大规模预训练模型
2. 集成多个模型预测结果，提高鲁棒性
3. 引入外部知识库增强中医文本理解
4. 设计更专业的领域特定预训练任务
