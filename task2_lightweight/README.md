# 轻量级中药处方推荐模型

本目录包含基于BERT的轻量级中药处方推荐模型，用于任务2：根据病例信息推荐中药处方。

## 模型特点

- **轻量级设计**：使用BERT作为基础模型，通过减少参数和简化结构，适合计算资源有限的环境
- **多头注意力增强**：使用自注意力机制增强文本特征理解
- **药物相关性表示**：学习中药之间的关联性，提高处方的合理性
- **证型疾病信息整合**：可选地结合任务1的证型和疾病预测，提高处方准确性
- **双重处方生成策略**：同时使用阈值法和Top-K法，确保处方中药数量合理

## 目录结构

```
task2_lightweight/
├── models.py          # 模型定义
├── utils.py           # 数据处理和工具函数
├── train.py           # 模型训练脚本
├── predict.py         # 模型预测脚本
├── run_training.sh    # 训练运行脚本
├── run_prediction.sh  # 预测运行脚本
└── README.md          # 说明文档
```

## 安装依赖

```bash
pip install torch transformers sklearn matplotlib tqdm numpy
```

## 数据准备

1. 训练数据: JSON格式，包含病例信息和处方标签
2. 中药列表: 文本文件，每行一个中药名称
3. 验证数据(可选): 用于监控训练过程和模型性能

## 使用方法

### 训练模型

```bash
# 使用默认参数训练
bash run_training.sh

# 或自定义参数
python train.py \
  --train_file ../data/train.json \
  --val_file ../data/val.json \
  --herbs_file ../data/herbs_list.txt \
  --model_name bert-base-chinese \
  --batch_size 16 \
  --max_length 512 \
  --learning_rate 3e-5 \
  --epochs 5 \
  --output_dir ./output
```

### 预测中药处方

```bash
# 使用默认参数预测
bash run_prediction.sh

# 或自定义参数
python predict.py \
  --test_file ../data/test.json \
  --output_file ./predictions.json \
  --model_path ./output/best_model.pth \
  --herbs_list ./output/herbs_list.json \
  --mlb_file ./output/mlb.json \
  --model_name bert-base-chinese \
  --model_type basic \
  --max_length 512 \
  --threshold 0.5 \
  --top_k 15
```

## 模型架构

### 基础模型 (TCMHerbsLightModel)

1. **BERT编码层**: 提取文本特征
2. **多头自注意力层**: 增强文本理解
3. **前馈神经网络**: 非线性特征转换
4. **药物注意力机制**: 计算文本对不同中药的关注度
5. **药物分类器**: 预测每种中药的使用概率

### 增强模型 (TCMHerbsWithSyndromeDiseaseModel)

1. **基础模型**: 继承基础模型的所有功能
2. **证型疾病特征投影**: 将证型和疾病信息映射到特征空间
3. **门控融合机制**: 整合文本、证型和疾病特征
4. **强化分类器**: 综合多源信息进行药物预测

## 训练与评估

- **损失函数**: 二元交叉熵(BCEWithLogitsLoss)
- **优化器**: AdamW
- **学习率调度**: 线性预热+线性衰减
- **评估指标**:
  - Micro-F1: 所有样本所有中药的平均F1分数
  - Sample-F1: 每个样本的F1分数平均值
  - Micro-Jaccard: 所有样本所有中药的平均Jaccard相似度
  - Sample-Jaccard: 每个样本的Jaccard相似度平均值

## 实验结果

本模型相比于原始的大型语言模型微调方法，具有以下优势：

1. **显存占用低**: 适合GPU内存有限的环境
2. **训练速度快**: 从几小时缩短到数十分钟
3. **模型大小小**: 仅需约400MB存储空间
4. **性能保持**: 在测试集上能达到可接受的预测性能

## 注意事项

1. 请确保处理好中药名称的标准化，避免因药名变体导致的匹配问题
2. 调整`top_k`参数以控制推荐处方的中药数量
3. 对于大规模数据集，可考虑使用更大的批次大小和更多轮次训练

## 进一步改进方向

1. 添加中药知识图谱信息
2. 实现药物协同过滤机制
3. 考虑中药剂量和用法的预测
4. 集成多个模型的预测结果 