"""任务2配置文件：中药处方推荐模型参数设置"""

class Config:
    # 数据路径
    train_file = "../dataset/TCM-TBOSD-train.json"
    val_file = "../dataset/TCM-TBOSD-train.json"  # 使用训练集的一部分作为验证集
    test_file = "../dataset/TCM-TBOSD-test-B.json"  # B榜测试集
    task1_predictions = "../task1/predictions.json"  # 任务1的预测结果
    herbs_file = "../data/herbs.txt"  # 中药列表
    
    # 数据分割
    val_ratio = 0.1  # 验证集比例，从训练集中划分10%作为验证
    
    # 输出路径
    output_dir = "../models/task2"
    submission_file = "../outputs/task2_submission_B.json"  # 修改为B榜提交文件名
    
    # 模型参数
    base_model = "../models/pretrained/Qwen/Qwen-7B"  # 本地模型路径
    load_in_4bit = True        # 4bit量化训练
    lora_r = 16                # LoRA秩
    lora_alpha = 32            # LoRA缩放参数
    lora_dropout = 0.05        # LoRA dropout
    target_modules = ["c_attn", "c_proj", "w1", "w2"]  # 适用于早期Qwen模型的目标模块
    
    # 训练参数
    max_length = 4096          # 最大序列长度
    learning_rate = 2e-5       # 学习率
    weight_decay = 0.01        # 权重衰减
    num_epochs = 6             # 训练轮次
    per_device_train_batch_size = 1  # 训练批次大小
    gradient_accumulation_steps = 8  # 梯度累积步数
    warmup_ratio = 0.1         # 预热比例
    
    # 生成参数
    num_beams = 1              # 束搜索数量
    top_p = 0.85               # 采样top_p
    temperature = 0.7          # 温度参数
    num_candidates = 2         # 每个样本生成候选数
    min_herbs = 10             # 最小中药数量
    max_herbs = 15             # 最大中药数量
    
    # 后处理参数
    herbs_threshold = 0.5      # 中药选择阈值
    
    # 评估参数
    eval_steps = 20            # 评估间隔
    save_steps = 50            # 保存间隔
    
    # 输出参数
    seed = 42                  # 随机种子 