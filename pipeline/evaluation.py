import os
import pandas as pd
import torch
from dotenv import load_dotenv
import lm_eval
from google import genai
import re

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------
# OLMES Base Evaluation (Stages 1, 2, 3)
# ---------------------------------------------------------
def evaluate_base_model(model_path, tokenizer, stage_name):
    print(f"--- Starting OLMES Base Evaluation for {stage_name} ---")
    
    # The 10 tasks defined in the OLMES paper
    olmes_tasks = [
        "arc_challenge", "arc_easy", "boolq", "hellaswag", 
        "mmlu", "openbookqa", "piqa", "social_iqa", 
        "winogrande", "commonsense_qa"
    ]
    
    # Initialize lm_eval model wrapper
    # Assuming model_path is a HuggingFace model object or path
    model_args = f"pretrained={model_path}" if isinstance(model_path, str) else model_path
    
    print(f"Running lm_eval for tasks: {olmes_tasks}")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=olmes_tasks,
        num_fewshot=5, # OLMES standardizes on 5-shot for most tasks
        batch_size="auto",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Extract metrics
    metrics = []
    for task_name, task_results in results['results'].items():
        # OLMES takes the max of MCF (acc) and CF (acc_norm) where applicable
        acc = task_results.get('acc,none', task_results.get('acc'))
        acc_norm = task_results.get('acc_norm,none', task_results.get('acc_norm'))
        
        best_score = max(acc if acc is not None else 0, acc_norm if acc_norm is not None else 0)
        
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
# OLMo 3 Post-Training Evaluation (Stages 4, 5, 6)
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
        
        # Extract score using regex
        score_match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", output)
        score = float(score_match.group(1)) if score_match else 0.0
        return score, output
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return 0.0, str(e)

def evaluate_post_trained_model(model, tokenizer, stage_name):
    print(f"--- Starting OLMo 3 Post-Training Evaluation for {stage_name} ---")
    
    # 1. Standard Generative Tasks (Math, Code, Reasoning) via lm_eval
    post_train_tasks = ["gsm8k", "mathqa", "ifeval"]
    
    model_args = f"pretrained={model}" if isinstance(model, str) else model
    print(f"Running lm_eval for generative tasks: {post_train_tasks}")
    
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=post_train_tasks,
        num_fewshot=0,
        batch_size="auto",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    metrics = []
    for task_name, task_results in results['results'].items():
        metrics.append({
            "Stage": stage_name,
            "Task": task_name,
            "Score": task_results.get('exact_match,none', task_results.get('acc,none', 0))
        })

    # 2. Chat & Instruction Following Evaluation (LLM-as-a-Judge via Gemini)
    print("Running LLM-as-a-Judge Chat Evaluation using Gemini 3.5 Flash...")
    
    # Sample prompts representing AlpacaEval / WildChat style queries
    chat_prompts = [
        "Explain the theory of relativity to a 5-year-old.",
        "Write a Python script to reverse a linked list.",
        "What are the main causes of the French Revolution?"
    ]
    
    chat_scores = []
    for prompt in chat_prompts:
        # Format prompt using the tokenizer's chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate response (Assuming 'model' is a loaded HF model here; adjust if passing paths)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.6, top_p=0.95)
        
        response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Judge with Gemini
        score, reasoning = gemini_llm_judge(prompt, response_text)
        chat_scores.append(score)
    
    avg_chat_score = sum(chat_scores) / len(chat_scores) if chat_scores else 0
    metrics.append({
        "Stage": stage_name,
        "Task": "alpaca_eval_gemini_judge",
        "Score": avg_chat_score
    })
    
    # Generate Report
    df = pd.DataFrame(metrics)
    report_path = f"{stage_name}_PostTrain_report.csv"
    df.to_csv(report_path, index=False)
    print(f"Saved Post-Training Evaluation Report to {report_path}\n")
    return df