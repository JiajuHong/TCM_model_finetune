import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import os
import sys
import math
import logging

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 现在可以直接导入项目内的模块
from utils.config import Config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SelfAttentionLayer(nn.Module):
    """自注意力层，用于增强特征提取"""
    
    def __init__(self, hidden_size, dropout_prob=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        
        # 自注意力投影层
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout和输出投影
        self.dropout = nn.Dropout(dropout_prob)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # 确保输入是3D张量
        original_shape = hidden_states.shape
        if len(original_shape) == 4:
            # 如果是4D张量，将中间两个维度展平
            logger.info(f"Reshaping 4D tensor of shape {original_shape} to 3D")
            batch_size, dim1, dim2, hidden_dim = original_shape
            hidden_states = hidden_states.reshape(batch_size, dim1 * dim2, hidden_dim)
        elif len(original_shape) == 2:
            # 如果是2D张量[batch_size, hidden_size]，则添加序列维度
            hidden_states = hidden_states.unsqueeze(1)
        elif len(original_shape) != 3:
            logger.error(f"Unexpected tensor shape in SelfAttentionLayer: {original_shape}")
            # 如果不是2D、3D或4D，则返回输入
            return hidden_states
            
        # 计算自注意力
        query = self.query_proj(hidden_states)
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)
        
        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 调整attention_mask的形状以匹配hidden_states的序列长度
            if attention_mask.shape[1] != hidden_states.shape[1]:
                logger.warning(f"Attention mask shape {attention_mask.shape} doesn't match hidden states shape {hidden_states.shape}")
                # 如果mask长度不匹配，创建一个新的与hidden_states长度匹配的mask
                attention_mask = torch.ones((hidden_states.shape[0], hidden_states.shape[1]), device=hidden_states.device)
            
            # 转换attention_mask形状为[batch_size, 1, 1, seq_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 将0转换为-10000.0，1保持不变，实现掩码效果
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask
        
        # Softmax并应用dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 获取加权上下文向量
        context = torch.matmul(attention_probs, value)
        context = self.output_proj(context)
        
        return context


class LabelAttention(nn.Module):
    """标签注意力机制，用于聚焦特定证型/疾病的特征"""
    
    def __init__(self, hidden_size, num_labels):
        super(LabelAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        # 标签注意力层
        self.feature_transform = nn.Linear(hidden_size, hidden_size//2)
        self.attention_weights = nn.Linear(hidden_size//2, num_labels)
        
    def forward(self, hidden_states):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        returns: [batch_size, num_labels, hidden_size]
        """
        # 确保输入是3D张量
        original_shape = hidden_states.shape
        if len(original_shape) == 4:
            # 如果是4D张量，将中间两个维度展平
            logger.info(f"Reshaping 4D tensor of shape {original_shape} to 3D")
            batch_size, dim1, dim2, hidden_dim = original_shape
            hidden_states = hidden_states.reshape(batch_size, dim1 * dim2, hidden_dim)
        elif len(original_shape) == 2:
            # 如果是2D张量[batch_size, hidden_size]，则添加序列维度
            hidden_states = hidden_states.unsqueeze(1)
        elif len(original_shape) != 3:
            logger.error(f"Unexpected tensor shape: {original_shape}")
            # 如果不是2D、3D或4D，则返回一个安全的默认结果
            batch_size = original_shape[0]
            return torch.zeros((batch_size, self.num_labels, self.hidden_size), device=hidden_states.device)
        
        # 特征转换
        transformed_features = torch.tanh(self.feature_transform(hidden_states))  # [batch_size, seq_len, hidden_size//2]
        
        # 计算每个标签的注意力权重
        attention_weights = self.attention_weights(transformed_features)  # [batch_size, seq_len, num_labels]
        attention_weights = attention_weights.transpose(1, 2)  # [batch_size, num_labels, seq_len]
        
        # 应用softmax获取归一化的注意力权重
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 计算加权和
        label_features = torch.bmm(attention_weights, hidden_states)  # [batch_size, num_labels, hidden_size]
        
        return label_features


class TCMJointModel(nn.Module):
    """改进的证型-疾病联合建模"""
    
    def __init__(self, config=None):
        super(TCMJointModel, self).__init__()
        self.config = Config() if config is None else config
        
        # 加载预训练模型
        self.model_config = AutoConfig.from_pretrained(
            self.config.PRETRAINED_MODEL,
            cache_dir=self.config.CACHE_DIR
        )
        self.base_model = AutoModel.from_pretrained(
            self.config.PRETRAINED_MODEL,
            config=self.model_config,
            cache_dir=self.config.CACHE_DIR
        )
        
        # 获取模型隐藏层大小
        self.hidden_size = self.model_config.hidden_size
        
        # 添加自注意力层增强特征提取
        self.self_attention = SelfAttentionLayer(self.hidden_size)
        
        # 标签注意力机制
        self.syndrome_attention = LabelAttention(self.hidden_size, self.config.NUM_SYNDROME_LABELS)
        
        # 证型分类器
        self.syndrome_classifier = nn.Linear(self.hidden_size, 1)
        
        # 证型特征转换，用于疾病预测
        self.syndrome_feature_transform = nn.Linear(self.config.NUM_SYNDROME_LABELS, self.hidden_size//2)
        
        # 疾病分类器（结合证型特征）
        self.disease_classifier = nn.Linear(self.hidden_size + self.hidden_size//2, self.config.NUM_DISEASE_LABELS)
        
        # Dropout层
        self.dropout = nn.Dropout(0.1)
        
        # 用于简单线性分类的备用层
        self.simple_syndrome_classifier = nn.Linear(self.hidden_size, self.config.NUM_SYNDROME_LABELS)
        self.simple_disease_classifier = nn.Linear(self.hidden_size, self.config.NUM_DISEASE_LABELS)
        
    def forward(self, input_ids, attention_mask, syndrome_labels=None, disease_labels=None):
        """前向传播"""
        device = input_ids.device
        
        # 确保所有标签都在正确的设备上
        if syndrome_labels is not None:
            syndrome_labels = syndrome_labels.to(device)
        if disease_labels is not None:
            disease_labels = disease_labels.to(device)
        
        # 获取基础模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取序列表示
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 检查序列输出的维度
        if len(sequence_output.shape) != 3:
            logger.warning(f"Unexpected sequence_output shape: {sequence_output.shape}")
        
        try:
            # 应用自注意力增强特征
            enhanced_sequence = self.self_attention(sequence_output, attention_mask)
            
            # 证型预测：应用标签注意力机制
            syndrome_features = self.syndrome_attention(enhanced_sequence)  # [batch_size, num_syndrome_labels, hidden_size]
            
            # 每个证型的分类logits
            syndrome_logits = self.syndrome_classifier(self.dropout(syndrome_features)).squeeze(-1)  # [batch_size, num_syndrome_labels]
            
            # CLS token的特征
            cls_feature = sequence_output[:, 0, :]  # [batch_size, hidden_size]
            
            # 如果在推理阶段，使用预测的证型概率
            if syndrome_labels is None:
                syndrome_probs = torch.sigmoid(syndrome_logits)
            else:
                # 在训练阶段，使用真实标签（教师强制）以提供更稳定的学习信号
                syndrome_probs = syndrome_labels
            
            # 转换证型特征用于疾病预测
            syndrome_context = self.syndrome_feature_transform(syndrome_probs)  # [batch_size, hidden_size//2]
            
            # 结合CLS特征和证型特征进行疾病预测
            combined_features = torch.cat([cls_feature, syndrome_context], dim=1)  # [batch_size, hidden_size + hidden_size//2]
            disease_logits = self.disease_classifier(self.dropout(combined_features))  # [batch_size, num_disease_labels]
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            logger.error(f"Sequence output shape: {sequence_output.shape}")
            
            # 使用简单的线性方法
            # 确保我们有一个有效的CLS特征，无论输入形状如何
            if len(sequence_output.shape) == 3 and sequence_output.shape[1] > 0:
                cls_feature = sequence_output[:, 0, :]
            elif len(sequence_output.shape) == 4 and sequence_output.shape[1] > 0 and sequence_output.shape[2] > 0:
                cls_feature = sequence_output[:, 0, 0, :]
            else:
                # 创建一个随机初始化的特征向量
                cls_feature = torch.zeros((input_ids.size(0), self.hidden_size), device=device)
            
            # 使用简单的线性层进行预测
            syndrome_logits = self.simple_syndrome_classifier(cls_feature)
            disease_logits = self.simple_disease_classifier(cls_feature)
        
        # 计算损失
        loss = None
        if syndrome_labels is not None and disease_labels is not None:
            # 确保所有张量在同一设备上
            syndrome_logits = syndrome_logits.to(device)
            disease_logits = disease_logits.to(device)
            
            # 证型损失（多标签）
            syndrome_loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=self.syndrome_weights.to(device) if hasattr(self, 'syndrome_weights') else None
            )
            syndrome_loss = syndrome_loss_fct(syndrome_logits, syndrome_labels)
            
            # 疾病损失（单标签但使用BCE）
            disease_loss_fct = nn.BCEWithLogitsLoss()
            disease_loss = disease_loss_fct(disease_logits, disease_labels)
            
            # 总损失为证型损失和疾病损失的加权和
            loss = syndrome_loss + disease_loss
        
        return {
            'loss': loss,
            'syndrome_logits': syndrome_logits,
            'disease_logits': disease_logits
        }
    
    def set_syndrome_weights(self, weights):
        """设置证型类权重"""
        if weights is not None:
            self.syndrome_weights = weights
            logger.info(f"Set syndrome weights: {self.syndrome_weights}")


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