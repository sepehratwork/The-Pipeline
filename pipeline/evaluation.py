import os
import pandas as pd
import torch
import requests
from dotenv import load_dotenv
import lm_eval
from lm_eval.models.huggingface import HFLM  # Wrap HF models correctly
from google import genai
import re
from transformers import AutoModelForCausalLM

# Load environment variables from .env file
load_dotenv()

# Enable arbitrary Python execution safely for evaluation tasks (e.g., HumanEval)
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# ---------------------------------------------------------
# Robust Metric Extraction Helper
# ---------------------------------------------------------
def get_best_metric(task_results):
    """
    Extracts the primary/best performance metric score from a task results dictionary.
    Handles variations in metric naming across different versions of lm_eval.
    """
    # Priority list of metric keys representing accuracy, exact match, or pass rate
    metric_keys = [
        'acc,none', 'acc', 
        'acc_norm,none', 'acc_norm', 
        'exact_match,none', 'exact_match',
        'pass@1,none', 'pass@1',
        'acc_pmi,none', 'acc_pmi',
        'acc_mutual_info,none',
        'f1,none', 'f1'
    ]
    
    for key in metric_keys:
        val = task_results.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
                
    # Fallback: scan all keys for common matching terms
    for k, v in task_results.items():
        if any(substring in k.lower() for substring in ['acc', 'exact', 'match', 'score', 'pass']):
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
                
    # Ultimate fallback: return the first numerical value found
    for k, v in task_results.items():
        try:
            return float(v)
        except (ValueError, TypeError):
            pass
            
    return 0.0

# ---------------------------------------------------------
# OLMES Base Evaluation (Stages 1, 2, 3)
# ---------------------------------------------------------
def evaluate_base_model(model_path, tokenizer, stage_name):
    print(f"--- Starting OLMES Base Evaluation for {stage_name} ---")
    
    # Complete 10 standard tasks defined in the OLMES paper
    olmes_tasks = [
        "arc_challenge", "arc_easy", "boolq", "hellaswag", 
        "mmlu", "openbookqa", "piqa", "social_iqa", 
        "winogrande", "commonsense_qa"
    ]
    
    # Initialize the model wrapper and provide the pre-configured tokenizer
    # HFLM automatically supports both string paths and preloaded model objects
    print("Wrapping model in HFLM wrapper with pre-configured tokenizer...")
    lm_obj = HFLM(
        pretrained=model_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Running lm_eval for tasks: {olmes_tasks}")
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=olmes_tasks,
        num_fewshot=5,  # OLMES standardizes on 5-shot for most tasks
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Extract metrics
    metrics = []
    for task_name, task_results in results['results'].items():
        best_score = get_best_metric(task_results)
        acc = task_results.get('acc,none', task_results.get('acc'))
        acc_norm = task_results.get('acc_norm,none', task_results.get('acc_norm'))
        
        metrics.append({
            "Stage": stage_name,
            "Task": task_name,
            "Accuracy": acc,
            "Accuracy_Norm": acc_norm,
            "OLMES_Score (Max)": best_score
        })
    
    # Generate Report
    df = pd.DataFrame(metrics)
    report_path = f"{stage_name}_OLMES_report.csv"
    df.to_csv(report_path, index=False)
    print(f"Saved Base Evaluation Report to {report_path}\n")
    return df

# ---------------------------------------------------------
# LLM-as-a-Judge Functions
# ---------------------------------------------------------
def gemini_llm_judge(prompt, model_response):
    """Uses Gemini 3.5 Flash to judge open-ended chat responses."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    generation_config = {
        'max_output_tokens': 65536,
        'thinking_level': 'medium',
    }
    
    judge_prompt = (
        "You are an impartial judge evaluating an AI assistant's response.\n"
        f"User Prompt: {prompt}\n"
        f"AI Response: {model_response}\n\n"
        "Evaluate the response for helpfulness, accuracy, and instruction-following. "
        "Provide a brief reasoning, then score the response from 1 to 10. "
        "Format your output exactly as: 'SCORE: X' at the very end."
    )
    
    try:
        interaction = client.interactions.create(
            model='models/gemini-3.5-flash',
            input=judge_prompt,
            generation_config=generation_config,
        )
        output = interaction.output_text
        
        score_match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", output)
        score = float(score_match.group(1)) if score_match else 0.0
        return score, output
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return 0.0, str(e)

def cloudflare_llm_judge(prompt, model_response):
    """Uses Cloudflare API (glm-5.2) to judge open-ended chat responses."""
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    auth_token = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
    
    judge_prompt = (
        "You are an impartial judge evaluating an AI assistant's response.\n"
        f"User Prompt: {prompt}\n"
        f"AI Response: {model_response}\n\n"
        "Evaluate the response for helpfulness, accuracy, and instruction-following. "
        "Provide a brief reasoning, then score the response from 1 to 10. "
        "Format your output exactly as: 'SCORE: X' at the very end."
    )
    
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/zai-org/glm-5.2"
    headers = {"Authorization": f"Bearer {auth_token}"}
    payload = {
        "messages": [
            {"role": "system", "content": "You are a strict and impartial judge."},
            {"role": "user", "content": judge_prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        output = result.get("result", {}).get("response", "")
        
        score_match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", output)
        score = float(score_match.group(1)) if score_match else 0.0
        return score, output
    except Exception as e:
        print(f"Cloudflare API Error: {e}")
        return 0.0, str(e)

# ---------------------------------------------------------
# OLMo 3 Post-Training Evaluation (Stages 4, 5, 6)
# ---------------------------------------------------------
def evaluate_post_trained_model(model, tokenizer, stage_name, judge_api="gemini"):
    """
    Evaluates post-trained models on mathematical, instruction following, and coding capabilities.
    judge_api can be either 'gemini' or 'cloudflare'.
    """
    print(f"--- Starting OLMo 3 Post-Training Evaluation for {stage_name} ---")
    print(f"Using LLM Judge API: {judge_api.upper()}")
    
    # Complete core post-training tasks (including MATH and code generation)
    post_train_tasks = ["gsm8k", "hendrycks_math", "ifeval", "humaneval"]
    
    print("Wrapping model in HFLM wrapper with pre-configured tokenizer...")
    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Running lm_eval for generative tasks: {post_train_tasks}")
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=post_train_tasks,
        num_fewshot=0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    metrics = []
    for task_name, task_results in results['results'].items():
        score = get_best_metric(task_results)
        metrics.append({
            "Stage": stage_name,
            "Task": task_name,
            "Score": score
        })

    # 2. Chat & Instruction Following Evaluation (LLM-as-a-Judge)
    print(f"Running LLM-as-a-Judge Chat Evaluation using {judge_api}...")
    
    chat_prompts = [
        "Explain the theory of relativity to a 5-year-old.",
        "Write a Python script to reverse a linked list.",
        "What are the main causes of the French Revolution?"
    ]
    
    # Ensure model is ready for local generation (handles string path and PyTorch objects cleanly)
    generation_model = model
    if isinstance(model, str):
        print(f"Loading model from path '{model}' for chat generation...")
        generation_model = AutoModelForCausalLM.from_pretrained(
            model, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
    chat_scores = []
    for prompt in chat_prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(generation_model.device)
        with torch.no_grad():
            outputs = generation_model.generate(**inputs, max_new_tokens=512, temperature=0.6, top_p=0.95)
        
        response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Route to the selected Judge API
        if judge_api == "gemini":
            score, reasoning = gemini_llm_judge(prompt, response_text)
        elif judge_api == "cloudflare":
            score, reasoning = cloudflare_llm_judge(prompt, response_text)
        else:
            raise ValueError("Invalid judge_api. Choose 'gemini' or 'cloudflare'.")
            
        chat_scores.append(score)
    
    avg_chat_score = sum(chat_scores) / len(chat_scores) if chat_scores else 0
    metrics.append({
        "Stage": stage_name,
        "Task": f"alpaca_eval_{judge_api}_judge",
        "Score": avg_chat_score
    })
    
    # Generate Report
    df = pd.DataFrame(metrics)
    report_path = f"{stage_name}_PostTrain_report.csv"
    df.to_csv(report_path, index=False)
    print(f"Saved Post-Training Evaluation Report to {report_path}\n")
    return df