# 任务2: 中药处方推荐模型

本目录包含针对中医辨证辨病及中药处方生成测评比赛中的任务2（中药处方推荐）的优化实现。

## 方案概述

基于大语言模型（Qwen-2.5-7B-Instruct）的优化微调方案，通过提示工程和数据增强提升中药处方推荐准确率。

### 主要特点

1. **优化提示模板**：
   - 集成辨证辨病结果
   - 添加证型特定用药提示
   - 引入经典方剂框架
   - 支持少样本学习

2. **数据增强**：
   - 药物共现关系分析
   - 证型-药物映射
   - 疾病-药物映射

3. **多候选生成**：
   - 并行生成多个处方方案
   - 基于评分函数选择最优方案

4. **后处理优化**：
   - 规范化药物名称
   - 约束处方数量
   - 自动纠错和筛选

## 文件说明

- `config.py`: 配置参数和路径
- `prompts.py`: 提示模板设计
- `data_utils.py`: 数据处理和增强
- `utils.py`: 评估和后处理工具
- `train.py`: 模型训练脚本
- `eval.py`: 模型评估脚本
- `predict.py`: 测试集预测脚本
- `run.sh`: 一键运行脚本

## 使用方法

### 环境准备

需要安装以下依赖：
```
transformers==4.36.2
peft==0.7.1
accelerate==0.25.0
bitsandbytes==0.41.1
datasets==2.14.6
torch==2.1.2
```

### 训练模型

```bash
./run.sh
```

或分步执行：

```bash
# 训练
python train.py

# 评估
python eval.py

# 预测
python predict.py
```

### 配置修改

在`config.py`中可以修改以下配置：
- 数据路径和输出路径
- 模型参数
- 训练参数
- 生成参数

## 实现细节

### 提示工程优化

```
任务：作为经验丰富的中医师，根据患者[基本信息],[主诉],[症状],[中医望闻切诊],[病史],[体格检查],[辅助检查]等信息，为患者开具合适的中药处方。

# 中医辨证结果
证型判断：【证型1】【证型2】(如有)
疾病判断：【疾病名】

# 处方原则
1. 必须严格从下方药物列表中选择，不得添加列表外药物
2. 根据"君臣佐使"原则组方，兼顾疾病和证型特点
3. 药物数量应控制在10-15味之间
4. 输出格式为逗号分隔的药物列表，不包含剂量

对于【证型】常用药物: 【药物列表】
参考方剂：【经典方剂名称】

# 可选药物列表
[草药]: {herbs_list}

[基本信息]:患者性别为{gender},年龄为{age},职业为{job},婚姻为{marriage},发病节气为{disease_time}
[主诉]:{chief_complaint}
[症状]:{symptom}
[中医望闻切诊]:{tcm_examination}
[病史]:{history}
[体格检查]:{physical_examination}
[辅助检查]:{auxiliary_examination}

请根据以上信息，列出推荐的中药处方:
```

### 评估指标

基于Jaccard相似系数、F1分数和平均药物数量匹配度的综合评分：

```
task2_score = (jaccard + f1 + avg_herbs) / 3
```

## 参考资料

- [中医辨证辨病及中药处方生成测评比赛官网](https://tianchi.aliyun.com/competition/entrance/532301)
- [任务2 Baseline实现](../baseline/baseline/task2/)
- [全参数微调vs. LoRA](https://huggingface.co/blog/lora)
- [提示工程指南](https://www.promptingguide.ai/) 