"""中药处方推荐模型预测脚本：用于生成测试集预测结果"""

import os
import json
from tqdm import tqdm
import logging
from config import Config
from data_utils import load_herbs_list, load_dataset, load_task1_predictions, prepare_evaluation_data
from utils import select_best_candidate, prepare_submission
from eval import load_model_and_tokenizer, generate_prescription

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("task2_predict.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def predict(config, model_path=None, data_path=None, herbs_file=None, output_file=None, base_model_path=None, task1_pred_path=None):
    """为测试集生成预测结果"""
    # 使用传入的路径或配置文件中的路径
    model_path = model_path or config.output_dir
    data_path = data_path or config.test_file
    herbs_file = herbs_file or config.herbs_file
    output_file = output_file or config.submission_file
    base_model_path = base_model_path or config.base_model
    task1_pred_path = task1_pred_path or config.task1_predictions
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path, base_model_path)
    
    # 加载中药列表
    herbs_list_path = os.path.join(model_path, "herbs_list.json")
    if os.path.exists(herbs_list_path):
        logger.info(f"Loading herbs list from {herbs_list_path}")
        with open(herbs_list_path, "r", encoding="utf-8") as f:
            herbs_list = json.load(f)
    else:
        logger.info(f"Loading herbs list from {herbs_file}")
        herbs_list = load_herbs_list(herbs_file)
    
    herbs_set = set(herbs_list)
    
    # 加载任务1预测结果（如果有）
    task1_predictions = None
    if os.path.exists(task1_pred_path):
        logger.info(f"Loading task1 predictions from {task1_pred_path}")
        task1_predictions = load_task1_predictions(task1_pred_path)
    
    # 加载测试数据
    logger.info(f"Loading test data from {data_path}")
    test_data = load_dataset(data_path, herbs_list)
    
    # 准备测试数据
    test_output_file = os.path.join(os.path.dirname(model_path), "test_data.jsonl")
    test_samples = prepare_evaluation_data(test_data, herbs_list, test_output_file, task1_predictions)
    
    # 预测结果
    predictions = {}
    
    # 为每个样本生成处方
    logger.info("Generating prescriptions for test samples")
    for item in tqdm(test_samples, desc="Predicting"):
        sample_id = item["id"]
        prompt = item["input"]
        
        # 生成候选处方
        candidates = generate_prescription(model, tokenizer, prompt, config)
        
        # 后处理并选择最佳候选
        best_candidate = select_best_candidate(
            candidates, 
            None,  # 测试集没有参考处方 
            herbs_set,
            config.min_herbs,
            config.max_herbs
        )
        
        # 保存预测结果
        predictions[sample_id] = best_candidate
    
    # 将预测结果准备为提交格式并保存
    logger.info(f"Preparing submission file {output_file}")
    submission = prepare_submission(predictions, output_file)
    
    logger.info(f"Prediction completed with {len(predictions)} samples")
    return predictions

if __name__ == "__main__":
    config = Config()
    predict(config) 