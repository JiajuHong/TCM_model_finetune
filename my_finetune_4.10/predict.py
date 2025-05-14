import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import sys

from models import TCMSyndromeDiseaseModel, TCMHerbsModel
from utils import load_test_data, generate_submission, load_herbs, SYNDROME_CLASSES, DISEASE_CLASSES

def predict_syndrome_disease(args):
    """Predict syndromes and diseases"""
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        
        # Load test data
        test_data = load_test_data(args.test_file, tokenizer)
        print(f"Loaded {len(test_data)} test samples")
        
        # Load model
        model = TCMSyndromeDiseaseModel(model_name=args.model_name)
        model.load_state_dict(torch.load(args.syndrome_disease_model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Predictions
        predictions = {}
        
        with torch.no_grad():
            for item in tqdm(test_data, desc="Predicting syndromes and diseases"):
                # Prepare input
                encoding = tokenizer(
                    item['text'],
                    truncation=True,
                    max_length=args.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Model prediction
                syndrome_logits, disease_logits = model(input_ids, attention_mask)
                
                # Use threshold for syndromes
                syndrome_probs = torch.sigmoid(syndrome_logits).cpu().numpy()[0]
                syndrome_indices = np.where(syndrome_probs > args.syndrome_threshold)[0]
                
                # If no syndromes predicted, select the top one
                if len(syndrome_indices) == 0:
                    syndrome_indices = [np.argmax(syndrome_probs)]
                
                # Use argmax for disease
                disease_index = np.argmax(disease_logits.cpu().numpy()[0])
                
                # Convert to syndrome and disease names
                predicted_syndrome = SYNDROME_CLASSES[syndrome_indices[0]]  # Use only the top syndrome
                predicted_disease = DISEASE_CLASSES[disease_index]
                
                # According to the example: ["气虚血瘀证", "胸痹心痛病"]
                predictions[item['ID']] = [predicted_syndrome, predicted_disease]
        
        return predictions
    except Exception as e:
        print(f"Error predicting syndromes and diseases: {e}")
        return {}

def predict_herbs(args):
    """Predict herb prescriptions"""
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        
        # Load herbs list
        herbs_list = load_herbs()
        print(f"Total herbs: {len(herbs_list)}")
        
        # Load test data
        test_data = load_test_data(args.test_file, tokenizer)
        print(f"Loaded {len(test_data)} test samples")
        
        # Load model
        model = TCMHerbsModel(model_name=args.model_name, num_herbs=len(herbs_list))
        model.load_state_dict(torch.load(args.herbs_model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Predictions
        predictions = {}
        
        with torch.no_grad():
            for item in tqdm(test_data, desc="Predicting herb prescriptions"):
                # Prepare input
                encoding = tokenizer(
                    item['text'],
                    truncation=True,
                    max_length=args.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Model prediction
                herbs_logits = model(input_ids, attention_mask)
                
                # Get herb prediction results
                herbs_probs = torch.sigmoid(herbs_logits).cpu().numpy()[0]
                
                # Select top N herbs
                top_indices = np.argsort(herbs_probs)[::-1][:args.top_herbs]
                
                # Filter out herbs with probability below threshold
                selected_indices = [i for i in top_indices if herbs_probs[i] > args.herbs_threshold]
                
                # If too few herbs predicted, supplement with top herbs
                if len(selected_indices) < args.min_herbs:
                    # Add up to minimum number
                    remaining = args.min_herbs - len(selected_indices)
                    for idx in top_indices:
                        if idx not in selected_indices and remaining > 0:
                            selected_indices.append(idx)
                            remaining -= 1
                
                # Convert to herb names - list of herb strings
                predicted_herbs = [herbs_list[i] for i in selected_indices]
                
                # According to the example: ['黄连', '炙甘草', '牡蛎', ...]
                predictions[item['ID']] = predicted_herbs
        
        return predictions
    except Exception as e:
        print(f"Error predicting herb prescriptions: {e}")
        return {}

def main():
    """Main function, set default parameters and run prediction tasks"""
    # Default parameters
    class Args:
        def __init__(self):
            # Data parameters
            self.test_file = 'data/TCM-TBOSD-test-B.json'
            self.output_file = 'submission.json'
            
            # Model parameters
            self.model_name = 'hfl/chinese-bert-wwm-ext'
            self.syndrome_disease_model_path = './models/best_syndrome_disease_model.pth'
            self.herbs_model_path = './models/best_herbs_model.pth'
            self.max_length = 512
            
            # Prediction parameters
            self.syndrome_threshold = 0.4
            self.disease_threshold = 0.5
            self.herbs_threshold = 0.3
            self.top_herbs = 20
            self.min_herbs = 10
            
            # Task selection
            self.task = 'both'  # 'syndrome_disease', 'herbs', 'both'
    
    args = Args()
    
    # Check if test file exists
    if not os.path.exists(args.test_file):
        print(f"Error: Test file {args.test_file} does not exist")
        return
    
    # Check if model files exist
    if args.task in ['syndrome_disease', 'both'] and not os.path.exists(args.syndrome_disease_model_path):
        print(f"Error: Syndrome-disease model file {args.syndrome_disease_model_path} does not exist")
        print("Skipping task 1")
        args.task = 'herbs' if args.task == 'both' else args.task
    
    if args.task in ['herbs', 'both'] and not os.path.exists(args.herbs_model_path):
        print(f"Error: Herbs model file {args.herbs_model_path} does not exist")
        print("Skipping task 2")
        args.task = 'syndrome_disease' if args.task == 'both' else args.task
    
    # Run predictions
    task1_predictions = {}
    task2_predictions = {}
    
    if args.task in ['syndrome_disease', 'both']:
        print("Predicting syndromes and diseases...")
        task1_predictions = predict_syndrome_disease(args)
    
    if args.task in ['herbs', 'both']:
        print("Predicting herb prescriptions...")
        task2_predictions = predict_herbs(args)
    
    # Generate submission file
    generate_submission(task1_predictions, task2_predictions, args.output_file)
    print(f"Prediction completed, results saved to {args.output_file}")

if __name__ == "__main__":
    main() 