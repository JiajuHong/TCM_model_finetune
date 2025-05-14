#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轻量级中药处方推荐模型一键运行脚本"""

import os
import sys
import logging
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[
        logging.FileHandler("task2_lightweight_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_train():
    """运行训练流程"""
    logger.info("开始训练中药处方推荐模型...")
    try:
        # 确保输出目录存在
        os.makedirs("./output", exist_ok=True)
        
        # 调用训练脚本 - 使用直接参数方式
        from train import train_herbs_model
        train_herbs_model(
            train_file="../dataset/TCM-TBOSD-train.json",
            herbs_file="../data/herbs.txt",
            output_dir="./output",
            epochs=5,
            batch_size=16,
            model_name="hfl/chinese-roberta-wwm-ext"
        )
        
        logger.info("训练完成！")
        return True
    except Exception as e:
        logger.error(f"训练过程发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_predict():
    """运行预测流程"""
    logger.info("开始进行中药处方推荐预测...")
    try:
        # 调用预测脚本
        import predict
        predict.main()
        
        logger.info("预测完成！")
        return True
    except Exception as e:
        logger.error(f"预测过程发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="一键运行中药处方推荐模型")
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all", 
                      help="运行模式: 训练(train), 预测(predict), 或两者(all)")
    parser.add_argument("--data_dir", type=str, default="../data", 
                      help="数据目录")
    parser.add_argument("--train_file", type=str, default="../dataset/TCM-TBOSD-train.json",
                      help="训练数据文件路径")
    parser.add_argument("--herbs_file", type=str, default="../data/herbs.txt",
                      help="中药列表文件路径")
    parser.add_argument("--output_dir", type=str, default="./output",
                      help="输出目录")
    parser.add_argument("--epochs", type=int, default=5,
                      help="训练轮数")
    
    args = parser.parse_args()
    
    # 设置工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 根据模式运行
    if args.mode in ["train", "all"]:
        # 如果指定了参数，使用参数覆盖默认值
        if hasattr(run_train, "__code__"):
            fn_args = run_train.__code__.co_varnames[:run_train.__code__.co_argcount]
            train_args = {key: getattr(args, key) for key in fn_args if hasattr(args, key)}
        
        success = run_train()
        if not success and args.mode == "all":
            logger.error("训练失败，不继续执行预测.")
            return
    
    if args.mode in ["predict", "all"]:
        run_predict()

if __name__ == "__main__":
    main() 