"""
Evaluation pipeline for automatic judging of model responses.
- Loads processed dataset (data/processed_csprob.csv)
- Runs each question through selected models (Llama-3-8B, Mistral-7B, Qwen-7B)
- Uses judge models (Qwen, Llama from /media/12TB/shared/datasets/) to verify responses
- Saves results to evaluation/results.csv
"""


import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import infer_auto_device_map, dispatch_model
import gc
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from datetime import datetime

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '../.env')))
HF_TOKEN = os.getenv('HF_TOKEN')

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed_csprob.csv'))
RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/results.csv'))
LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/evaluation.log'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
MODEL_CONFIGS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    # "qwen-7b": "Qwen/Qwen-7B"
}

JUDGE_MODELS = {
    "qwen-judge": "Qwen/Qwen2.5-72B-Instruct"
}

# General resource optimization settings
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Adjust based on VRAM, can be tuned
torch.set_float32_matmul_precision('high')

def load_model(model_id, device="cuda", quantization_bits=8):
    """
    Load model with specified quantization.
    Args:
        quantization_bits: 4, 8, or None (for FP16)
    """
    cache_dir = "/media/12TB/shared/models"
    trust_remote_code = False
    if "Qwen" in str(model_id) or "qwen" in str(model_id):
        trust_remote_code = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=HF_TOKEN,
        trust_remote_code=trust_remote_code
    )
    # Use quantization for memory efficiency
    from transformers import BitsAndBytesConfig
    if quantization_bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            cache_dir=cache_dir,
            token=HF_TOKEN,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config
        )
    elif quantization_bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            cache_dir=cache_dir,
            token=HF_TOKEN,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config
        )
    else:
        # No quantization - FP16 for best quality
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
            token=HF_TOKEN,
            trust_remote_code=trust_remote_code
        )
    # Do NOT call model.to(device) when using device_map="auto"
    return model, tokenizer

def generate_response(model, tokenizer, question, device="cuda"):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512)
    # Free up memory after generation
    gc.collect()
    torch.cuda.empty_cache()
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def judge_response(judge_model, judge_tokenizer, question, answer, response, device="cuda"):
    # Simplified prompt for single overall score considering multiple criteria
    prompt = (
        f"You are an expert judge. Evaluate the model's response considering correctness, completeness, relevancy, and clarity.\n"
        f"Question: {question}\n"
        f"Reference Answer: {answer}\n"
        f"Model Response: {response}\n\n"
        f"Provide a single overall score (0-1 scale): "
    )
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = judge_model.generate(**inputs, max_new_tokens=256)
    judge_output = judge_tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract overall score from judge output
    import re
    match = re.search(r"([01](?:\.\d+)?|0?\.\d+)", judge_output)
    overall_score = float(match.group(1)) if match else None
    logger.info(f"Score: {overall_score}")
    
    return overall_score

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run a small batch test (first 2 rows)')
    parser.add_argument('--append', action='store_true', help='Append new model columns to existing results')
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    
    # Load existing results if append mode
    if args.append and os.path.exists(RESULTS_PATH):
        results_df = pd.read_csv(RESULTS_PATH)
        logger.info(f"Loaded {len(results_df)} existing results. Will add new model columns.")
    else:
        # Initialize results DataFrame with base columns
        results_df = df[['Q_ID', 'Domain', 'Difficulty', 'Question', 'Answer']].copy()
        results_df.rename(columns={'Answer': 'Reference_Answer'}, inplace=True)
    
    logger.info("Loading models...")
    # Load evaluation models WITH 4-bit quantization for speed and memory savings
    models = {name: load_model(mid, TORCH_DEVICE, quantization_bits=4) for name, mid in MODEL_CONFIGS.items()}
    # Load judge models WITH 4-bit quantization for speed and memory (they're much larger)
    judge_models = {name: load_model(path, TORCH_DEVICE, quantization_bits=4) for name, path in JUDGE_MODELS.items()}
    logger.info(f"Loaded {len(models)} evaluation model(s) and {len(judge_models)} judge model(s)")

    if args.test:
        df = df.head(2)
        results_df = results_df.head(2)
        logger.info('Running test mode: evaluating first 2 rows only.')

    # Progress bar for all questions
    total_evaluations = len(df) * len(models)
    pbar = tqdm(total=total_evaluations, desc="Evaluation Progress")
    
    # Get the first judge model (assuming we use one judge for all)
    judge_name, (judge_model, judge_tokenizer) = list(judge_models.items())[0]
    
    # Evaluate each model and add columns
    for mname, (model, tokenizer) in models.items():
        # Get model display name with quantization info
        model_id = MODEL_CONFIGS[mname]
        model_short_name = model_id.split('/')[-1]  # e.g., "Meta-Llama-3-8B"
        response_col = f"{model_short_name}(4bit)_Response"
        score_col = f"{model_short_name}(4bit)_Score"
        
        # Skip if columns already exist (append mode)
        if args.append and score_col in results_df.columns:
            logger.info(f"Skipping {mname} - already evaluated")
            pbar.update(len(df))
            continue
        
        responses = []
        scores = []
        
        # Process in batches
        for start in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[start:start+BATCH_SIZE]
            for idx, row in batch.iterrows():
                q_id = row['Q_ID']
                question = row['Question']
                answer = row['Answer']
                
                pbar.set_description(f"Evaluating {mname} on Q{q_id}")
                logger.info(f"Model: {mname}, Q_ID: {q_id}")
                
                # Generate response
                response = generate_response(model, tokenizer, question, TORCH_DEVICE)
                responses.append(response)
                
                # Judge response
                overall_score = judge_response(judge_model, judge_tokenizer, question, answer, response, TORCH_DEVICE)
                scores.append(overall_score)
                
                pbar.update(1)
            
            # Free up memory after each batch
            gc.collect()
            torch.cuda.empty_cache()
        
        # Add columns to results_df
        results_df[response_col] = responses
        results_df[score_col] = scores
        
        # Save after each model to avoid data loss
        results_df.to_csv(RESULTS_PATH, index=False)
        logger.info(f"Added {mname} results and saved to {RESULTS_PATH}")
    
    pbar.close()
    logger.info(f"Evaluation complete! Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
