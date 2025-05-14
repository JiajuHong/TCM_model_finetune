"""中药处方推荐任务B榜预测脚本：直接处理test-B.json生成B榜提交结果"""

import os
import subprocess
import json
import logging
from tqdm import tqdm
from config import Config
from data_utils import load_herbs_list, load_dataset, load_task1_predictions, prepare_evaluation_data
from utils import select_best_candidate, prepare_submission
from eval import load_model_and_tokenizer, generate_prescription

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("task2_predict_B.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def predict_B(config):
    """B榜预测流程：先生成任务1预测，再生成任务2预测"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(config.submission_file), exist_ok=True)
    
    # 1. 先运行任务1的预测
    logger.info("Step 1: Running task1 prediction...")
    task1_command = [
        "python", "../task1/predict.py",
        # 可以在这里添加任务1的预测参数，如有需要
    ]
    subprocess.run(task1_command, check=True)
    logger.info(f"Task1 prediction completed. Results saved to {config.task1_predictions}")
    
    # 2. 加载模型和分词器
    logger.info("Step 2: Loading task2 model...")
    model, tokenizer = load_model_and_tokenizer(config.output_dir, config.base_model)
    
    # 3. 加载中药列表
    logger.info("Step 3: Loading herbs list...")
    herbs_list_path = os.path.join(config.output_dir, "herbs_list.json")
    if os.path.exists(herbs_list_path):
        with open(herbs_list_path, "r", encoding="utf-8") as f:
            herbs_list = json.load(f)
    else:
        herbs_list = load_herbs_list(config.herbs_file)
    herbs_set = set(herbs_list)
    
    # 4. 加载任务1预测结果
    logger.info("Step 4: Loading task1 predictions...")
    task1_predictions = load_task1_predictions(config.task1_predictions)
    if not task1_predictions:
        logger.warning("Task1 predictions not found or empty! This will affect performance.")
    
    # 5. 加载B榜测试数据
    logger.info("Step 5: Loading test-B data...")
    test_data = load_dataset(config.test_file, herbs_list)
    
    # 6. 准备评估数据
    logger.info("Step 6: Preparing test data with task1 predictions...")
    test_output_file = os.path.join(os.path.dirname(config.output_dir), "test_B_data.jsonl")
    test_samples = prepare_evaluation_data(test_data, herbs_list, test_output_file, task1_predictions)
    
    # 7. 进行预测
    logger.info("Step 7: Generating prescriptions for B-board samples...")
    predictions = {}
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
    
    # 8. 准备提交格式并保存
    logger.info(f"Step 8: Preparing B-board submission file {config.submission_file}")
    submission = prepare_submission(predictions, config.submission_file)
    
    logger.info(f"B-board prediction completed with {len(predictions)} samples")
    logger.info(f"Submission file saved to {config.submission_file}")
    
    return predictions

if __name__ == "__main__":
    config = Config()
    predict_B(config) 