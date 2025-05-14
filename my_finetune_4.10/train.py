import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from models import TCMSyndromeDiseaseModel, TCMHerbsModel, TCMJointModel
from utils import load_data, load_herbs, SYNDROME_CLASSES, DISEASE_CLASSES

def plot_training_curves(metrics, title, save_path):
    """Plot training curves with multiple metrics"""
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot loss curves
    axes[0].plot(epochs, metrics['train_loss'], 'b-', marker='o', markersize=8, linewidth=2, label='Training Loss')
    if 'val_loss' in metrics and len(metrics['val_loss']) == len(epochs):
        axes[0].plot(epochs, metrics['val_loss'], 'r-', marker='s', markersize=8, linewidth=2, label='Validation Loss')
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(fontsize=10)
    
    # Define colors and markers for evaluation metrics
    colors = ['g', 'purple', 'orange', 'c', 'm', 'y']
    markers = ['d', '^', 'v', '<', '>', 'p', '*']
    
    # Filter metrics for plotting
    eval_metrics = [k for k in metrics.keys() 
                  if k not in ['train_loss', 'val_loss'] 
                  and len(metrics[k]) == len(epochs)]
    
    # Plot evaluation metrics
    for i, metric_name in enumerate(eval_metrics):
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        axes[1].plot(epochs, metrics[metric_name], 
                    color=colors[color_idx], 
                    marker=markers[marker_idx], 
                    markersize=8,
                    linewidth=2,
                    label=metric_name.replace('_', ' ').title())
    
    axes[1].set_title('Evaluation Metrics', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Only add legend if there are metrics to show
    if eval_metrics:
        axes[1].legend(fontsize=10)
    
    # Set overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)  # Make space for the overall title
    
    # Save figure with high DPI for better quality
    plt.savefig(save_path, dpi=300)
    print(f"Training curves saved to {save_path}")
    plt.close(fig)

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

def train_syndrome_disease_model(args):
    """训练证型疾病模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Model will be saved to: {os.path.abspath(args.output_dir)}")
        
        # 加载tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        
        # 加载数据
        train_loader, val_loader = load_data(
            args.train_file, 
            args.val_file, 
            tokenizer, 
            batch_size=args.batch_size,
            max_length=args.max_length,
            task_type="syndrome_disease"
        )
        
        if train_loader is None or len(train_loader) == 0:
            print(f"Error: Failed to load training data, please check path: {args.train_file}")
            return
            
        # 初始化模型
        model = TCMSyndromeDiseaseModel(model_name=args.model_name, dropout_rate=args.dropout_rate)
        model.to(device)
        
        # 定义损失函数和优化器
        syndrome_criterion = nn.BCEWithLogitsLoss()
        disease_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # 学习率调度器
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # 用于记录训练指标的字典
        metrics = {
            'train_loss': [],
            'train_syndrome_acc': [],
            'train_disease_acc': [],
            'train_combined_acc': []
        }
        
        # 如果有验证集，添加验证指标
        if val_loader is not None:
            metrics.update({
                'val_loss': [],
                'syndrome_f1': [],
                'disease_acc': [],
                'combined_score': []
            })
        
        # 训练循环
        best_score = 0.0
        save_flag = False  # 标记是否已保存模型
        
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            train_syndrome_correct = 0
            train_disease_correct = 0
            train_syndrome_total = 0
            train_disease_total = 0
            
            # 进度条
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for batch in progress_bar:
                # 将数据移到设备上
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                syndrome_labels = batch['syndrome_labels'].to(device)
                disease_labels = batch['disease_labels'].to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                syndrome_logits, disease_logits = model(input_ids, attention_mask)
                
                # 计算损失
                syndrome_loss = syndrome_criterion(syndrome_logits, syndrome_labels)
                disease_loss = disease_criterion(disease_logits, disease_labels.argmax(dim=1))
                loss = syndrome_loss + disease_loss
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # 优化器步进
                optimizer.step()
                scheduler.step()
                
                # 计算训练精度
                train_syndrome_preds = (torch.sigmoid(syndrome_logits) > 0.5).float()
                train_syndrome_correct += (train_syndrome_preds == syndrome_labels).float().sum().item()
                train_syndrome_total += syndrome_labels.numel()
                
                train_disease_preds = torch.argmax(disease_logits, dim=1)
                train_disease_correct += (train_disease_preds == disease_labels.argmax(dim=1)).sum().item()
                train_disease_total += disease_labels.size(0)
                
                # 记录损失
                train_loss += loss.item()
                
                # 计算当前批次的准确率
                batch_syndrome_acc = (train_syndrome_preds == syndrome_labels).float().mean().item()
                batch_disease_acc = (train_disease_preds == disease_labels.argmax(dim=1)).float().mean().item()
                batch_combined_acc = (batch_syndrome_acc + batch_disease_acc) / 2
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'synd_acc': f'{batch_syndrome_acc:.4f}',
                    'dis_acc': f'{batch_disease_acc:.4f}'
                })
            
            # 计算平均训练损失和精度
            train_loss /= len(train_loader)
            train_syndrome_acc = train_syndrome_correct / train_syndrome_total if train_syndrome_total > 0 else 0
            train_disease_acc = train_disease_correct / train_disease_total if train_disease_total > 0 else 0
            train_combined_acc = (train_syndrome_acc + train_disease_acc) / 2
            
            # 记录训练指标
            metrics['train_loss'].append(train_loss)
            metrics['train_syndrome_acc'].append(train_syndrome_acc)
            metrics['train_disease_acc'].append(train_disease_acc)
            metrics['train_combined_acc'].append(train_combined_acc)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, "
                  f"Syndrome Acc: {train_syndrome_acc:.4f}, Disease Acc: {train_disease_acc:.4f}, "
                  f"Combined Acc: {train_combined_acc:.4f}")
            
            # 验证
            if val_loader is not None:
                val_metrics = evaluate_syndrome_disease_model(model, val_loader, device, syndrome_criterion, disease_criterion)
                print(f"Validation: Loss: {val_metrics['loss']:.4f}, Syndrome F1: {val_metrics['syndrome_f1']:.4f}, "
                      f"Disease Acc: {val_metrics['disease_acc']:.4f}")
                
                # 保存指标
                metrics['val_loss'].append(val_metrics['loss'])
                metrics['syndrome_f1'].append(val_metrics['syndrome_f1'])
                metrics['disease_acc'].append(val_metrics['disease_acc'])
                current_score = val_metrics['syndrome_f1'] + val_metrics['disease_acc']
                metrics['combined_score'].append(current_score)
                
                # 只保存最佳模型
                if current_score > best_score:
                    best_score = current_score
                    # 保存最佳模型
                    model_path = os.path.join(args.output_dir, 'best_syndrome_disease_model.pth')
                    try:
                        torch.save(model.state_dict(), model_path)
                        print(f"New best model saved with score: {best_score:.4f} to {os.path.abspath(model_path)}")
                        save_flag = True  # 标记已保存模型
                    except Exception as e:
                        print(f"Error saving model: {e}")
            else:
                # 如果没有验证集，使用训练精度作为评估标准
                current_score = train_combined_acc
                if current_score > best_score:
                    best_score = current_score
                    model_path = os.path.join(args.output_dir, 'best_syndrome_disease_model.pth')
                    try:
                        torch.save(model.state_dict(), model_path)
                        print(f"New best model saved with training accuracy: {best_score:.4f} to {os.path.abspath(model_path)}")
                        save_flag = True
                    except Exception as e:
                        print(f"Error saving model: {e}")
            
            # 每个epoch都保存一次当前模型
            try:
                model_path = os.path.join(args.output_dir, f'syndrome_disease_model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Saved epoch {epoch+1} model to {os.path.abspath(model_path)}")
            except Exception as e:
                print(f"Error saving epoch model: {e}")
        
        # 如果没有保存最佳模型，则保存最终模型
        if not save_flag:
            try:
                model_path = os.path.join(args.output_dir, 'final_syndrome_disease_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Final model saved to {os.path.abspath(model_path)}")
            except Exception as e:
                print(f"Error saving final model: {e}")
        
        # 绘制训练曲线
        if len(metrics['train_loss']) > 0:
            try:
                plot_training_curves(
                    metrics,
                    "Syndrome and Disease Classification Training Curves",
                    os.path.join(args.output_dir, "syndrome_disease_training_curves.png")
                )
            except Exception as e:
                print(f"Error plotting training curves: {e}")
        
        print("Training completed. Model saved.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except Exception as e:
        print(f"Error: An exception occurred during syndrome and disease model training - {e}")
        print(f"Exception details: {str(e)}")
        return

def evaluate_syndrome_disease_model(model, data_loader, device, syndrome_criterion, disease_criterion):
    """评估证型疾病模型"""
    model.eval()
    total_loss = 0.0
    all_syndrome_logits = []
    all_syndrome_labels = []
    all_disease_logits = []
    all_disease_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            syndrome_labels = batch['syndrome_labels'].to(device)
            disease_labels = batch['disease_labels'].to(device)
            
            # 前向传播
            syndrome_logits, disease_logits = model(input_ids, attention_mask)
            
            # 计算损失
            syndrome_loss = syndrome_criterion(syndrome_logits, syndrome_labels)
            disease_loss = disease_criterion(disease_logits, disease_labels.argmax(dim=1))
            loss = syndrome_loss + disease_loss
            total_loss += loss.item()
            
            # 收集预测结果
            all_syndrome_logits.extend(syndrome_logits.cpu().numpy())
            all_syndrome_labels.extend(syndrome_labels.cpu().numpy())
            all_disease_logits.extend(disease_logits.cpu().numpy())
            all_disease_labels.extend(disease_labels.cpu().numpy())
    
    # 转换为numpy数组
    all_syndrome_logits = np.array(all_syndrome_logits)
    all_syndrome_labels = np.array(all_syndrome_labels)
    all_disease_logits = np.array(all_disease_logits)
    all_disease_labels = np.array(all_disease_labels)
    
    # 为每个证型找到最优阈值
    syndrome_preds = np.zeros_like(all_syndrome_labels)
    for i in range(all_syndrome_logits.shape[1]):
        threshold = find_threshold_micro(all_syndrome_logits[:, i], all_syndrome_labels[:, i])
        syndrome_preds[:, i] = (all_syndrome_logits[:, i] >= threshold).astype(float)
    
    # 对疾病使用argmax
    disease_preds = np.argmax(all_disease_logits, axis=1)
    disease_true = np.argmax(all_disease_labels, axis=1)
    
    # 计算指标
    syndrome_f1 = f1_score(all_syndrome_labels, syndrome_preds, average='macro')
    disease_acc = (disease_preds == disease_true).mean()
    
    return {
        'loss': total_loss / len(data_loader),
        'syndrome_f1': syndrome_f1,
        'disease_acc': disease_acc
    }

def train_herbs_model(args):
    """训练中药处方模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Model will be saved to: {os.path.abspath(args.output_dir)}")
        
        # 加载中药列表
        herbs_list = load_herbs()
        print(f"Total herbs: {len(herbs_list)}")
        
        # 加载tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        
        # 加载数据
        train_loader, val_loader = load_data(
            args.train_file, 
            args.val_file, 
            tokenizer, 
            batch_size=args.batch_size,
            max_length=args.max_length,
            task_type="herbs",
            herbs_list=herbs_list
        )
        
        if train_loader is None or len(train_loader) == 0:
            print(f"Error: Failed to load training data, please check path: {args.train_file}")
            return
            
        # 初始化模型
        model = TCMHerbsModel(model_name=args.model_name, dropout_rate=args.dropout_rate, num_herbs=len(herbs_list))
        model.to(device)
        
        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # 学习率调度器
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # 用于记录训练指标的字典
        metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'train_precision': [],
            'train_recall': []
        }
        
        # 如果有验证集，添加验证指标
        if val_loader is not None:
            metrics.update({
                'val_loss': [],
                'herbs_f1': [],
                'herbs_jaccard': []
            })
        
        # 训练循环
        best_f1 = 0.0
        save_flag = False  # 标记是否已保存模型
        
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            epoch_preds = []
            epoch_labels = []
            
            # 进度条
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for batch in progress_bar:
                # 将数据移到设备上
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                herbs_labels = batch['herbs_labels'].to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                herbs_logits = model(input_ids, attention_mask)
                
                # 计算损失
                loss = criterion(herbs_logits, herbs_labels)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # 优化器步进
                optimizer.step()
                scheduler.step()
                
                # 计算训练精度
                herbs_preds = (torch.sigmoid(herbs_logits) > 0.5).float()
                
                # 收集预测和标签用于计算指标
                epoch_preds.extend(herbs_preds.cpu().numpy())
                epoch_labels.extend(herbs_labels.cpu().numpy())
                
                # 计算批次精度
                batch_accuracy = (herbs_preds == herbs_labels).float().mean().item()
                
                # 记录损失
                train_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'accuracy': f'{batch_accuracy:.4f}'
                })
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            
            # 计算训练集指标
            epoch_preds = np.array(epoch_preds)
            epoch_labels = np.array(epoch_labels)
            
            # 计算准确率、精确率和召回率
            accuracy_sum = 0
            precision_sum = 0
            recall_sum = 0
            sample_count = 0
            
            for i in range(len(epoch_labels)):
                pred_set = set(np.where(epoch_preds[i] == 1)[0])
                true_set = set(np.where(epoch_labels[i] == 1)[0])
                
                if len(true_set) == 0 and len(pred_set) == 0:
                    continue
                
                # 计算交集
                intersection = len(pred_set.intersection(true_set))
                
                # 精确率: 交集/预测集合大小
                precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
                precision_sum += precision
                
                # 召回率: 交集/真实集合大小
                recall = intersection / len(true_set) if len(true_set) > 0 else 0
                recall_sum += recall
                
                # 更新样本计数
                sample_count += 1
            
            # 避免除以零
            if sample_count > 0:
                precision = precision_sum / sample_count
                recall = recall_sum / sample_count
            else:
                precision = 0
                recall = 0
            
            # 计算Accuracy为样本级别的精度
            accuracy = ((epoch_preds == epoch_labels).sum() / epoch_labels.size).item()
            
            # 记录训练指标
            metrics['train_loss'].append(train_loss)
            metrics['train_accuracy'].append(accuracy)
            metrics['train_precision'].append(precision)
            metrics['train_recall'].append(recall)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # 验证
            if val_loader is not None:
                val_metrics = evaluate_herbs_model(model, val_loader, device, criterion)
                print(f"Validation: Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, "
                      f"Jaccard: {val_metrics['jaccard']:.4f}")
                
                # 保存指标
                metrics['val_loss'].append(val_metrics['loss'])
                metrics['herbs_f1'].append(val_metrics['f1'])
                metrics['herbs_jaccard'].append(val_metrics['jaccard'])
                
                # 只保存最佳模型
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    # 保存最佳模型
                    model_path = os.path.join(args.output_dir, 'best_herbs_model.pth')
                    try:
                        torch.save(model.state_dict(), model_path)
                        print(f"New best model saved with F1: {best_f1:.4f} to {os.path.abspath(model_path)}")
                        save_flag = True  # 标记已保存模型
                    except Exception as e:
                        print(f"Error saving model: {e}")
            else:
                # 如果没有验证集，使用训练F1作为标准
                train_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                if train_f1 > best_f1:
                    best_f1 = train_f1
                    model_path = os.path.join(args.output_dir, 'best_herbs_model.pth')
                    try:
                        torch.save(model.state_dict(), model_path)
                        print(f"New best model saved with training F1: {best_f1:.4f} to {os.path.abspath(model_path)}")
                        save_flag = True
                    except Exception as e:
                        print(f"Error saving model: {e}")
            
            # 每个epoch都保存一次当前模型
            try:
                model_path = os.path.join(args.output_dir, f'herbs_model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Saved epoch {epoch+1} model to {os.path.abspath(model_path)}")
            except Exception as e:
                print(f"Error saving epoch model: {e}")
        
        # 如果没有保存最佳模型，则保存最终模型
        if not save_flag:
            try:
                model_path = os.path.join(args.output_dir, 'final_herbs_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Final model saved to {os.path.abspath(model_path)}")
            except Exception as e:
                print(f"Error saving final model: {e}")
        
        # 绘制训练曲线
        if len(metrics['train_loss']) > 0:
            try:
                plot_training_curves(
                    metrics,
                    "Herbs Recommendation Training Curves",
                    os.path.join(args.output_dir, "herbs_training_curves.png")
                )
            except Exception as e:
                print(f"Error plotting training curves: {e}")
        
        print("Training completed. Model saved.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except Exception as e:
        print(f"Error: An exception occurred during herbs model training - {e}")
        print(f"Exception details: {str(e)}")
        return

def evaluate_herbs_model(model, data_loader, device, criterion):
    """评估中药处方模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            herbs_labels = batch['herbs_labels'].to(device)
            
            # 前向传播
            herbs_logits = model(input_ids, attention_mask)
            
            # 计算损失
            loss = criterion(herbs_logits, herbs_labels)
            total_loss += loss.item()
            
            # 预测
            herbs_preds = (torch.sigmoid(herbs_logits) > 0.5).float().cpu().numpy()
            all_preds.extend(herbs_preds)
            all_labels.extend(herbs_labels.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算评价指标
    precision_scores = []
    recall_scores = []
    jaccard_scores = []
    f1_scores = []
    
    for i in range(len(all_labels)):
        pred_set = set(np.where(all_preds[i] == 1)[0])
        true_set = set(np.where(all_labels[i] == 1)[0])
        
        # 计算交集
        intersection = len(pred_set.intersection(true_set))
        
        # 计算并集
        union = len(pred_set.union(true_set))
        
        # 精确率: 交集/预测集合大小
        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
        precision_scores.append(precision)
        
        # 召回率: 交集/真实集合大小
        recall = intersection / len(true_set) if len(true_set) > 0 else 0
        recall_scores.append(recall)
        
        # Jaccard相似度: 交集/并集
        jaccard = intersection / union if union > 0 else 0
        jaccard_scores.append(jaccard)
        
        # F1分数: 2 * (精确率 * 召回率) / (精确率 + 召回率)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return {
        'loss': total_loss / len(data_loader),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1': np.mean(f1_scores),
        'jaccard': np.mean(jaccard_scores)
    }

def train_joint_model(args):
    """训练联合模型，同时进行辨证辨病和药物推荐"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Model will be saved to: {os.path.abspath(args.output_dir)}")
        
        # 加载中药列表
        herbs_list = load_herbs()
        print(f"Total herbs: {len(herbs_list)}")
        
        # 加载tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        
        # 加载数据 - 使用辨证辨病的数据格式
        train_loader, val_loader = load_data(
            args.train_file, 
            args.val_file, 
            tokenizer, 
            batch_size=args.batch_size,
            max_length=args.max_length,
            task_type="joint",
            herbs_list=herbs_list
        )
        
        if train_loader is None or len(train_loader) == 0:
            print(f"Error: Failed to load training data, please check path: {args.train_file}")
            return
            
        # 初始化联合模型
        model = TCMJointModel(model_name=args.model_name, dropout_rate=args.dropout_rate, num_herbs=len(herbs_list))
        model.to(device)
        
        # 定义损失函数
        syndrome_criterion = nn.BCEWithLogitsLoss()
        disease_criterion = nn.CrossEntropyLoss()
        herbs_criterion = nn.BCEWithLogitsLoss()
        
        # 定义优化器
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # 学习率调度器
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # 用于记录训练指标的字典
        metrics = {
            'train_loss': [],
            'train_syndrome_acc': [],
            'train_disease_acc': [],
            'train_herbs_f1': [],
            'train_herbs_jaccard': []
        }
        
        # 如果有验证集，添加验证指标
        if val_loader is not None:
            metrics.update({
                'val_loss': [],
                'syndrome_f1': [],
                'disease_acc': [],
                'herbs_f1': [],
                'herbs_jaccard': [],
                'combined_score': []
            })
        
        # 训练循环
        best_score = 0.0
        save_flag = False  # 标记是否已保存模型
        
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            syndrome_preds_list = []
            syndrome_labels_list = []
            disease_preds_list = []
            disease_labels_list = []
            herbs_preds_list = []
            herbs_labels_list = []
            
            # 进度条
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for batch in progress_bar:
                # 将数据移到设备上
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                syndrome_labels = batch['syndrome_labels'].to(device)
                disease_labels = batch['disease_labels'].to(device)
                herbs_labels = batch['herbs_labels'].to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播 - 联合模式
                syndrome_logits, disease_logits, herbs_logits = model(input_ids, attention_mask, mode='joint')
                
                # 计算损失
                syndrome_loss = syndrome_criterion(syndrome_logits, syndrome_labels)
                disease_loss = disease_criterion(disease_logits, disease_labels.argmax(dim=1))
                herbs_loss = herbs_criterion(herbs_logits, herbs_labels)
                
                # 总损失 - 可以调整权重
                loss = syndrome_loss + disease_loss + herbs_loss
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # 优化器步进
                optimizer.step()
                scheduler.step()
                
                # 计算训练预测
                syndrome_preds = (torch.sigmoid(syndrome_logits) > 0.5).float()
                disease_preds = torch.argmax(disease_logits, dim=1)
                herbs_preds = (torch.sigmoid(herbs_logits) > 0.5).float()
                
                # 收集预测和标签
                syndrome_preds_list.extend(syndrome_preds.cpu().numpy())
                syndrome_labels_list.extend(syndrome_labels.cpu().numpy())
                disease_preds_list.extend(disease_preds.cpu().numpy())
                disease_labels_list.extend(disease_labels.argmax(dim=1).cpu().numpy())
                herbs_preds_list.extend(herbs_preds.cpu().numpy())
                herbs_labels_list.extend(herbs_labels.cpu().numpy())
                
                # 记录损失
                train_loss += loss.item()
                
                # 计算当前批次的指标
                batch_syndrome_acc = (syndrome_preds == syndrome_labels).float().mean().item()
                batch_disease_acc = (disease_preds == disease_labels.argmax(dim=1)).float().mean().item()
                batch_herbs_acc = (herbs_preds == herbs_labels).float().mean().item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'syn_acc': f'{batch_syndrome_acc:.4f}',
                    'dis_acc': f'{batch_disease_acc:.4f}',
                    'herb_acc': f'{batch_herbs_acc:.4f}'
                })
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            
            # 转换为numpy数组
            syndrome_preds_arr = np.array(syndrome_preds_list)
            syndrome_labels_arr = np.array(syndrome_labels_list)
            disease_preds_arr = np.array(disease_preds_list)
            disease_labels_arr = np.array(disease_labels_list)
            herbs_preds_arr = np.array(herbs_preds_list)
            herbs_labels_arr = np.array(herbs_labels_list)
            
            # 计算训练指标
            train_syndrome_f1 = f1_score(syndrome_labels_arr, syndrome_preds_arr, average='macro', zero_division=0)
            train_disease_acc = (disease_preds_arr == disease_labels_arr).mean()
            
            # 计算herbs指标
            herbs_precision_sum = 0
            herbs_recall_sum = 0
            herbs_jaccard_sum = 0
            herbs_f1_sum = 0
            sample_count = 0
            
            for i in range(len(herbs_labels_arr)):
                pred_set = set(np.where(herbs_preds_arr[i] == 1)[0])
                true_set = set(np.where(herbs_labels_arr[i] == 1)[0])
                
                if len(true_set) == 0 and len(pred_set) == 0:
                    continue
                
                # 计算交集
                intersection = len(pred_set.intersection(true_set))
                
                # 计算并集
                union = len(pred_set.union(true_set))
                
                # 精确率: 交集/预测集合大小
                precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
                herbs_precision_sum += precision
                
                # 召回率: 交集/真实集合大小
                recall = intersection / len(true_set) if len(true_set) > 0 else 0
                herbs_recall_sum += recall
                
                # Jaccard相似度: 交集/并集
                jaccard = intersection / union if union > 0 else 0
                herbs_jaccard_sum += jaccard
                
                # F1分数: 2 * (精确率 * 召回率) / (精确率 + 召回率)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                herbs_f1_sum += f1
                
                # 更新样本计数
                sample_count += 1
            
            # 避免除以零
            if sample_count > 0:
                train_herbs_precision = herbs_precision_sum / sample_count
                train_herbs_recall = herbs_recall_sum / sample_count
                train_herbs_jaccard = herbs_jaccard_sum / sample_count
                train_herbs_f1 = herbs_f1_sum / sample_count
            else:
                train_herbs_precision = 0
                train_herbs_recall = 0
                train_herbs_jaccard = 0
                train_herbs_f1 = 0
            
            # 记录训练指标
            metrics['train_loss'].append(train_loss)
            metrics['train_syndrome_acc'].append(train_syndrome_f1)
            metrics['train_disease_acc'].append(train_disease_acc)
            metrics['train_herbs_f1'].append(train_herbs_f1)
            metrics['train_herbs_jaccard'].append(train_herbs_jaccard)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, "
                  f"Syndrome F1: {train_syndrome_f1:.4f}, Disease Acc: {train_disease_acc:.4f}, "
                  f"Herbs F1: {train_herbs_f1:.4f}, Herbs Jaccard: {train_herbs_jaccard:.4f}")
            
            # 验证
            if val_loader is not None:
                val_metrics = evaluate_joint_model(
                    model, val_loader, device, 
                    syndrome_criterion, disease_criterion, herbs_criterion
                )
                print(f"Validation: Loss: {val_metrics['loss']:.4f}, "
                      f"Syndrome F1: {val_metrics['syndrome_f1']:.4f}, "
                      f"Disease Acc: {val_metrics['disease_acc']:.4f}, "
                      f"Herbs F1: {val_metrics['herbs_f1']:.4f}, "
                      f"Herbs Jaccard: {val_metrics['herbs_jaccard']:.4f}")
                
                # 保存指标
                metrics['val_loss'].append(val_metrics['loss'])
                metrics['syndrome_f1'].append(val_metrics['syndrome_f1'])
                metrics['disease_acc'].append(val_metrics['disease_acc'])
                metrics['herbs_f1'].append(val_metrics['herbs_f1'])
                metrics['herbs_jaccard'].append(val_metrics['herbs_jaccard'])
                
                # 计算综合得分
                current_score = (val_metrics['syndrome_f1'] + val_metrics['disease_acc'] + 
                                 val_metrics['herbs_f1'] + val_metrics['herbs_jaccard']) / 4
                metrics['combined_score'].append(current_score)
                
                # 只保存最佳模型
                if current_score > best_score:
                    best_score = current_score
                    # 保存最佳模型
                    model_path = os.path.join(args.output_dir, 'best_joint_model.pth')
                    try:
                        torch.save(model.state_dict(), model_path)
                        print(f"New best model saved with score: {best_score:.4f} to {os.path.abspath(model_path)}")
                        save_flag = True  # 标记已保存模型
                    except Exception as e:
                        print(f"Error saving model: {e}")
            else:
                # 如果没有验证集，使用训练综合得分
                current_score = (train_syndrome_f1 + train_disease_acc + 
                                train_herbs_f1 + train_herbs_jaccard) / 4
                if current_score > best_score:
                    best_score = current_score
                    model_path = os.path.join(args.output_dir, 'best_joint_model.pth')
                    try:
                        torch.save(model.state_dict(), model_path)
                        print(f"New best model saved with training score: {best_score:.4f} to {os.path.abspath(model_path)}")
                        save_flag = True
                    except Exception as e:
                        print(f"Error saving model: {e}")
            
            # 每个epoch都保存一次当前模型
            try:
                model_path = os.path.join(args.output_dir, f'joint_model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Saved epoch {epoch+1} model to {os.path.abspath(model_path)}")
            except Exception as e:
                print(f"Error saving epoch model: {e}")
        
        # 如果没有保存最佳模型，则保存最终模型
        if not save_flag:
            try:
                model_path = os.path.join(args.output_dir, 'final_joint_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Final model saved to {os.path.abspath(model_path)}")
            except Exception as e:
                print(f"Error saving final model: {e}")
        
        # 绘制训练曲线
        if len(metrics['train_loss']) > 0:
            try:
                plot_training_curves(
                    metrics,
                    "Joint Model Training Curves",
                    os.path.join(args.output_dir, "joint_model_training_curves.png")
                )
            except Exception as e:
                print(f"Error plotting training curves: {e}")
        
        print("Joint model training completed. Model saved.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except Exception as e:
        print(f"Error: An exception occurred during joint model training - {e}")
        print(f"Exception details: {str(e)}")
        return

def evaluate_joint_model(model, data_loader, device, syndrome_criterion, disease_criterion, herbs_criterion):
    """评估联合模型"""
    model.eval()
    total_loss = 0.0
    syndrome_preds_list = []
    syndrome_labels_list = []
    disease_preds_list = []
    disease_labels_list = []
    herbs_preds_list = []
    herbs_labels_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            syndrome_labels = batch['syndrome_labels'].to(device)
            disease_labels = batch['disease_labels'].to(device)
            herbs_labels = batch['herbs_labels'].to(device)
            
            # 前向传播
            syndrome_logits, disease_logits, herbs_logits = model(input_ids, attention_mask, mode='joint')
            
            # 计算损失
            syndrome_loss = syndrome_criterion(syndrome_logits, syndrome_labels)
            disease_loss = disease_criterion(disease_logits, disease_labels.argmax(dim=1))
            herbs_loss = herbs_criterion(herbs_logits, herbs_labels)
            loss = syndrome_loss + disease_loss + herbs_loss
            total_loss += loss.item()
            
            # 计算预测
            syndrome_preds = (torch.sigmoid(syndrome_logits) > 0.5).float().cpu().numpy()
            disease_preds = torch.argmax(disease_logits, dim=1).cpu().numpy()
            herbs_preds = (torch.sigmoid(herbs_logits) > 0.5).float().cpu().numpy()
            
            # 收集预测和标签
            syndrome_preds_list.extend(syndrome_preds)
            syndrome_labels_list.extend(syndrome_labels.cpu().numpy())
            disease_preds_list.extend(disease_preds)
            disease_labels_list.extend(disease_labels.argmax(dim=1).cpu().numpy())
            herbs_preds_list.extend(herbs_preds)
            herbs_labels_list.extend(herbs_labels.cpu().numpy())
    
    # 转换为numpy数组
    syndrome_preds_arr = np.array(syndrome_preds_list)
    syndrome_labels_arr = np.array(syndrome_labels_list)
    disease_preds_arr = np.array(disease_preds_list)
    disease_labels_arr = np.array(disease_labels_list)
    herbs_preds_arr = np.array(herbs_preds_list)
    herbs_labels_arr = np.array(herbs_labels_list)
    
    # 计算指标
    syndrome_f1 = f1_score(syndrome_labels_arr, syndrome_preds_arr, average='macro', zero_division=0)
    disease_acc = (disease_preds_arr == disease_labels_arr).mean()
    
    # 计算Jaccard相似度
    precision_scores = []
    recall_scores = []
    jaccard_scores = []
    f1_scores = []
    
    for i in range(len(herbs_labels_arr)):
        pred_set = set(np.where(herbs_preds_arr[i] == 1)[0])
        true_set = set(np.where(herbs_labels_arr[i] == 1)[0])
        
        # 计算交集
        intersection = len(pred_set.intersection(true_set))
        
        # 计算并集
        union = len(pred_set.union(true_set))
        
        # 精确率: 交集/预测集合大小
        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
        precision_scores.append(precision)
        
        # 召回率: 交集/真实集合大小
        recall = intersection / len(true_set) if len(true_set) > 0 else 0
        recall_scores.append(recall)
        
        # Jaccard相似度: 交集/并集
        jaccard = intersection / union if union > 0 else 0
        jaccard_scores.append(jaccard)
        
        # F1分数: 2 * (精确率 * 召回率) / (精确率 + 召回率)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return {
        'loss': total_loss / len(data_loader),
        'syndrome_f1': syndrome_f1,
        'disease_acc': disease_acc,
        'herbs_precision': np.mean(precision_scores),
        'herbs_recall': np.mean(recall_scores),
        'herbs_f1': np.mean(f1_scores),
        'herbs_jaccard': np.mean(jaccard_scores)
    }

def main():
    """主函数，设置默认参数并运行训练任务"""
    # 默认参数配置
    class Args:
        def __init__(self):
            # 数据参数
            self.train_file = 'data/TCM-TBOSD-train.json'
            self.val_file = 'data/TCM-TBOSD-test-A.json'
            self.output_dir = './models'
            
            # 模型参数
            self.model_name = 'hfl/chinese-bert-wwm-ext'
            self.max_length = 512
            self.dropout_rate = 0.2
            
            # 训练参数
            self.batch_size = 8
            self.epochs = 20
            self.learning_rate = 2e-5
            self.weight_decay = 0.01
            self.max_grad_norm = 1.0
            
            # 任务选择
            self.task = 'both'  # 'syndrome_disease', 'herbs', 'both', 'joint'
    
    args = Args()
    
    # 确保输出目录存在
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory created at: {os.path.abspath(args.output_dir)}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
    
    # 检查数据文件是否存在
    if not os.path.exists(args.train_file):
        print(f"Error: Training data file does not exist - {args.train_file}")
        return
        
    if args.val_file and not os.path.exists(args.val_file):
        print(f"Warning: Validation data file does not exist - {args.val_file}")
        print("No validation will be performed")
        args.val_file = None
    
    # 根据任务类型执行训练
    if args.task == 'syndrome_disease' or args.task == 'both':
        print("Starting syndrome and disease model training...")
        train_syndrome_disease_model(args)
    
    if args.task == 'herbs' or args.task == 'both':
        print("Starting herbs recommendation model training...")
        train_herbs_model(args)
        
    if args.task == 'joint':
        print("Starting joint model training...")
        train_joint_model(args)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 