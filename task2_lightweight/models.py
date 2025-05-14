import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel, BertConfig

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
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型参数"""
        nn.init.xavier_normal_(self.herbs_classifier.weight)
    
    def forward(self, input_ids, attention_mask):
        # 获取BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # 使用自注意力增强文本特征理解
        attn_output = self.self_attention(sequence_output, sequence_output, sequence_output, attention_mask)
        attn_output = self.dropout(attn_output)
        hidden_states = self.layer_norm1(sequence_output + attn_output)
        
        # 前馈神经网络
        ff_output = self.feed_forward(hidden_states)
        ff_output = self.dropout(ff_output)
        hidden_states = self.layer_norm2(hidden_states + ff_output)
        
        # 获取全局表示 - 使用[CLS]token或平均池化
        # 这里使用平均池化以捕获整个序列的信息
        global_repr = hidden_states.mean(dim=1)
        
        # 计算文本对药物的注意力
        herb_attention = self.herb_attention(global_repr)
        herb_attention = F.softmax(herb_attention, dim=-1)
        
        # 药物预测
        herbs_logits = self.herbs_classifier(global_repr)
        
        return herbs_logits

# 可选：使用证型和疾病信息辅助推荐的模型，如果有这些预测结果
class TCMHerbsWithSyndromeDiseaseModel(nn.Module):
    """结合证型和疾病信息的中药处方推荐模型"""
    def __init__(self, model_name="bert-base-chinese", dropout_rate=0.2, num_herbs=382, 
                 num_syndromes=10, num_diseases=4):
        super(TCMHerbsWithSyndromeDiseaseModel, self).__init__()
        # 基础文本编码
        self.text_encoder = TCMHerbsLightModel(
            model_name=model_name, 
            dropout_rate=dropout_rate, 
            num_herbs=num_herbs
        )
        
        # 证型和疾病信息整合
        self.syndrome_proj = nn.Linear(num_syndromes, self.text_encoder.hidden_size)
        self.disease_proj = nn.Linear(num_diseases, self.text_encoder.hidden_size)
        
        # 整合门控
        self.fusion = GatedFusion(self.text_encoder.hidden_size)
        
        # 最终分类器
        self.final_classifier = nn.Linear(self.text_encoder.hidden_size, num_herbs)
    
    def forward(self, input_ids, attention_mask, syndrome_probs=None, disease_probs=None):
        # 文本特征
        text_logits = self.text_encoder(input_ids, attention_mask)
        
        # 如果有证型和疾病信息，则整合
        if syndrome_probs is not None and disease_probs is not None:
            # 投影到隐藏空间
            syndrome_feats = self.syndrome_proj(syndrome_probs)
            disease_feats = self.disease_proj(disease_probs)
            
            # 整合证型和疾病特征
            sd_features = self.fusion(syndrome_feats, disease_feats)
            
            # 获取文本编码器的特征表示
            with torch.no_grad():
                text_features = self.text_encoder.bert(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                ).last_hidden_state.mean(dim=1)
            
            # 融合所有特征
            fused_features = self.fusion(text_features, sd_features)
            
            # 最终预测
            herbs_logits = self.final_classifier(fused_features)
            return herbs_logits
        else:
            # 如果没有证型疾病信息，直接返回文本特征预测结果
            return text_logits 