import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel, BertConfig
from utils import SYNDROME_CLASSES, DISEASE_CLASSES

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
        
        # 投影并重塑为多头格式 [batch_size, seq_len, num_heads, head_dim]
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
            # 扩展掩码至注意力分数的形状
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask.float()) * -10000.0
            scores = scores + extended_attention_mask
        
        # Softmax并应用dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算上下文向量
        context = torch.matmul(attention_weights, v)
        
        # 恢复原始形状 [batch_size, seq_len, hidden_size]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        # 最终线性投影
        output = self.output_proj(context)
        
        return output

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

class TCMSyndromeDiseaseModel(nn.Module):
    """增强的中医证型疾病分类模型"""
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
        
        # 难样本挖掘
        self.focal_gamma = 2.0
    
    def forward(self, input_ids, attention_mask):
        # 获取BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # 使用Transformer架构处理序列输出
        # 多头注意力 + 残差连接 + 层归一化
        attn_output = self.attention(sequence_output, sequence_output, sequence_output, attention_mask)
        attn_output = self.dropout(attn_output)
        hidden_states = self.layer_norm1(sequence_output + attn_output)
        
        # 前馈神经网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(hidden_states)
        ff_output = self.dropout(ff_output)
        hidden_states = self.layer_norm2(hidden_states + ff_output)
        
        # 提取证型和疾病特征
        syndrome_features = self.syndrome_feature(hidden_states)
        disease_features = self.disease_feature(pooled_output)
        
        # 证型分类 - 使用池化特征进行多标签分类
        # 对序列进行全局平均池化
        syndrome_pooled = syndrome_features.mean(dim=1)
        syndrome_logits = self.syndrome_classifier(self.dropout(syndrome_pooled))
        
        # 如果启用融合，将证型和疾病特征融合
        if self.use_fusion:
            disease_context = self.syndrome_disease_fusion(disease_features, syndrome_pooled)
        else:
            disease_context = disease_features
        
        # 疾病分类
        disease_logits = self.disease_classifier(self.dropout(disease_context))
        
        return syndrome_logits, disease_logits
    
    def compute_focal_loss(self, logits, targets, gamma=2.0):
        """计算Focal Loss以处理类别不平衡"""
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1-probs)
        focal_weight = (1 - pt) ** gamma
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = (focal_weight * loss).mean()
        return loss

class TCMHerbsModel(nn.Module):
    """增强的中药处方生成模型"""
    def __init__(self, model_name="hfl/chinese-bert-wwm-ext", dropout_rate=0.2, num_herbs=382):
        super(TCMHerbsModel, self).__init__()
        # BERT编码器
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.num_herbs = num_herbs
        
        # 多头注意力层
        self.self_attention = MultiHeadAttention(self.hidden_size, num_heads=8)
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
        
        # 药物共现表示学习
        self.herbs_emb = nn.Parameter(torch.randn(num_herbs, self.hidden_size))
        self.herbs_bias = nn.Parameter(torch.zeros(num_herbs))
        
        # 药物相关性注意力
        self.herb_attention = nn.Linear(self.hidden_size, num_herbs)
        
        # 用于药物推荐的多层分类器
        self.herbs_classifier1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.herbs_classifier2 = nn.Linear(self.hidden_size, num_herbs)
        
        # 药物互相影响的图卷积层
        self.gcn_weight = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型参数"""
        nn.init.xavier_uniform_(self.gcn_weight)
        nn.init.xavier_uniform_(self.herbs_emb)
    
    def build_herb_graph(self, hidden_states):
        """构建药物知识图谱"""
        # 计算药物之间的相似度
        herb_sim = torch.matmul(self.herbs_emb, self.herbs_emb.transpose(0, 1))
        # 归一化
        herb_sim = F.softmax(herb_sim, dim=-1)
        # 应用图卷积
        herb_features = torch.matmul(herb_sim, torch.matmul(self.herbs_emb, self.gcn_weight))
        return herb_features
    
    def forward(self, input_ids, attention_mask):
        # 获取BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 使用Transformer架构处理序列输出
        # 多头注意力 + 残差连接 + 层归一化
        attn_output = self.self_attention(sequence_output, sequence_output, sequence_output, attention_mask)
        attn_output = self.dropout(attn_output)
        hidden_states = self.layer_norm1(sequence_output + attn_output)
        
        # 前馈神经网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(hidden_states)
        ff_output = self.dropout(ff_output)
        hidden_states = self.layer_norm2(hidden_states + ff_output)
        
        # 获取整体表示
        global_repr = hidden_states.mean(dim=1)
        
        # 构建药物知识图谱
        herb_features = self.build_herb_graph(hidden_states)
        
        # 计算文本对药物的注意力
        herb_attention = self.herb_attention(global_repr)
        herb_attention = F.softmax(herb_attention, dim=-1)
        
        # 多层药物分类
        herbs_hidden = F.relu(self.herbs_classifier1(global_repr))
        herbs_hidden = self.dropout(herbs_hidden)
        herbs_logits = self.herbs_classifier2(herbs_hidden)
        
        # 融合药物知识图谱信息
        # 修复基于注意力的药物选择的维度问题
        batch_size = herb_attention.size(0)
        attended_herb_features = torch.matmul(herb_attention.view(batch_size, 1, -1), herb_features).squeeze(1)
        herbs_logits = herbs_logits + torch.matmul(attended_herb_features, self.herbs_emb.transpose(0, 1))
        
        # 应用药物偏置
        herbs_logits = herbs_logits + self.herbs_bias
        
        return herbs_logits

# 用于证型、疾病和药物之间关系建模的联合模型
class TCMJointModel(nn.Module):
    """中医综合诊断模型 - 联合辨证辨病和药物推荐"""
    def __init__(self, model_name="hfl/chinese-bert-wwm-ext", dropout_rate=0.2, num_herbs=382):
        super(TCMJointModel, self).__init__()
        # 共享BERT编码器
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        
        # 证型疾病分类器
        self.syndrome_disease_classifier = nn.ModuleDict({
            'syndrome_attn': nn.Linear(self.hidden_size, len(SYNDROME_CLASSES)),
            'syndrome_classifier': nn.Linear(self.hidden_size, len(SYNDROME_CLASSES)),
            'disease_classifier': nn.Linear(self.hidden_size, len(DISEASE_CLASSES))
        })
        
        # 药物推荐器
        self.herbs_recommender = nn.ModuleDict({
            'herbs_attn': nn.Linear(self.hidden_size, num_herbs),
            'herbs_classifier': nn.Linear(self.hidden_size, num_herbs)
        })
        
        # 模块间知识交互
        self.knowledge_transfer = nn.ModuleDict({
            'syndrome_to_herbs': nn.Linear(len(SYNDROME_CLASSES), num_herbs),
            'disease_to_herbs': nn.Linear(len(DISEASE_CLASSES), num_herbs),
            'fusion': GatedFusion(num_herbs)
        })
    
    def forward(self, input_ids, attention_mask, mode='joint'):
        # 获取BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        if mode == 'syndrome_disease' or mode == 'joint':
            # 对序列输出进行平均池化
            sequence_pooled = sequence_output.mean(dim=1)
            
            # 证型分类 - 多标签分类
            syndrome_logits = self.syndrome_disease_classifier['syndrome_classifier'](self.dropout(sequence_pooled))
            
            # 疾病分类
            disease_logits = self.syndrome_disease_classifier['disease_classifier'](self.dropout(pooled_output))
            
            # 仅证型疾病预测模式
            if mode == 'syndrome_disease':
                return syndrome_logits, disease_logits
        
        if mode == 'herbs' or mode == 'joint':
            # 基本药物推荐 - 使用序列的全局表示
            global_repr = sequence_output.mean(dim=1)
            
            # 药物推荐
            herbs_logits = self.herbs_recommender['herbs_classifier'](self.dropout(global_repr))
            
            # 如果是联合模式，考虑证型和疾病的影响
            if mode == 'joint':
                # 将证型和疾病知识转移到药物推荐
                syndrome_probs = torch.sigmoid(syndrome_logits)
                disease_probs = F.softmax(disease_logits, dim=-1)
                
                syndrome_influence = self.knowledge_transfer['syndrome_to_herbs'](syndrome_probs)
                disease_influence = self.knowledge_transfer['disease_to_herbs'](disease_probs)
                
                # 融合不同来源的影响
                knowledge_influence = self.knowledge_transfer['fusion'](syndrome_influence, disease_influence)
                
                # 结合到最终的药物预测
                herbs_logits = herbs_logits + knowledge_influence
                
                return syndrome_logits, disease_logits, herbs_logits
            
            # 仅药物推荐模式
            return herbs_logits 