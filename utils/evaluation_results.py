import os
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# Define paths
BASE_DIR = "/content/drive/MyDrive/Simulated/ModelsCheckpoints"
OUTPUT_CSV = "/content/drive/MyDrive/Simulated/evaluation_results.csv"

def get_task_cluster(task_name: str) -> str:
    """
    Categorizes tasks into OlmoBaseEval clusters matching Table 2 & 3 from OLMo 3 paper.
    """
    tn = task_name.lower()
    
    # 1. HeldOut Suite
    if any(k in tn for k in ['mmlu_pro', 'lbpp', 'deepmind_math', 'bbh']):
        return 'HeldOut'
        
    # 2. Math Suite
    if any(k in tn for k in ['gsm8k', 'gsm_symbolic', 'minerva_math', 'math500', 'math_500']):
        return 'Math'
        
    # 3. Code & FIM Suite
    if 'fim' in tn:
        return 'FIM'
    if any(k in tn for k in ['humaneval', 'mbpp', 'bigcodebench', 'ds_1000', 'ds1000', 'deepseek_leetcode', 'multipl_e']):
        return 'Code'
        
    # Check if multiple choice variant
    is_mc = ':mc' in tn or '_mc' in tn or 'gen2mc' in tn
    
    # 4. MC STEM Suite
    stem_keywords = [
        'arc_easy', 'arc_challenge', 'medmcqa', 'medqa', 'sciq',
        'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_physics',
        'computer_security', 'conceptual_physics', 'electrical_engineering',
        'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_mathematics',
        'high_school_physics', 'high_school_statistics', 'machine_learning'
    ]
    if any(k in tn for k in stem_keywords):
        return 'MC STEM'
        
    # 5. MC Non-STEM Suite (Includes Gen2MC tasks)
    non_stem_keywords = [
        'csqa', 'piqa', 'socialiqa',
        'formal_logic', 'european_history', 'us_history', 'world_history',
        'international_law', 'jurisprudence', 'logical_fallacies', 'moral_disputes',
        'moral_scenarios', 'philosophy', 'prehistory', 'professional_law',
        'world_religions', 'econometrics', 'geography', 'government_and_politics',
        'macroeconomics', 'microeconomics', 'psychology', 'human_sexuality',
        'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
        'anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine',
        'global_facts', 'human_aging', 'management', 'marketing', 'medical_genetics',
        'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine',
        'virology'
    ]
    if any(k in tn for k in non_stem_keywords):
        return 'MC Non-STEM'
        
    if is_mc and any(k in tn for k in ['coqa', 'drop', 'jeopardy', 'naturalqs', 'squad']):
        return 'MC Non-STEM'
        
    # 6. GenQA Suite (Short-form / RC generative tasks)
    genqa_keywords = ['hellaswag', 'winogrande', 'lambada', 'basic_skills', 'drop', 'jeopardy', 'naturalqs', 'squad', 'coqa']
    if any(k in tn for k in genqa_keywords):
        return 'GenQA'
        
    if 'mmlu' in tn:
        return 'MC Non-STEM'
        
    return 'Other'


def parse_checkpoint_metrics(metrics_files):
    """
    Parses all task-*-metrics.json files for a single checkpoint directory.
    """
    task_scores = {}
    cluster_scores = {}
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            task_name = data.get('task_name', os.path.basename(file_path).split('-metrics.json')[0])
            metrics = data.get('metrics', {})
            
            # Extract primary score
            score = metrics.get('primary_score', metrics.get('acc_raw', metrics.get('pass@1', None)))
            
            if score is None:
                continue
                
            # Normalize to 0-100 percentage scale if given as fraction 0.0-1.0
            if isinstance(score, (int, float)):
                if score <= 1.0 and score >= 0.0:
                    score = score * 100.0
                
            cluster = get_task_cluster(task_name)
            
            if cluster not in cluster_scores:
                cluster_scores[cluster] = []
            cluster_scores[cluster].append(score)
            task_scores[task_name] = score
            
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
            
    # Calculate macro-averages per cluster
    cluster_averages = {cluster: np.mean(scores) for cluster, scores in cluster_scores.items()}
    
    # Calculate overall main suite average (standard main clusters: MC STEM, MC Non-STEM, GenQA, Math, Code, FIM)
    main_clusters = ['MC STEM', 'MC Non-STEM', 'GenQA', 'Math', 'Code', 'FIM']
    avail_main_scores = [cluster_averages[c] for c in main_clusters if c in cluster_averages]
    
    if avail_main_scores:
        cluster_averages['Avg'] = np.mean(avail_main_scores)
    else:
        cluster_averages['Avg'] = np.nan
        
    return cluster_averages, task_scores


def aggregate_all_evaluations(base_dir, model_name, phase):
    """
    Walks through model checkpoints and aggregates evaluation tables.
    """
    records = []
    
    # Locate all directories containing *-metrics.json files
    metric_files = glob.glob(os.path.join(base_dir, f"{model_name}-{phase}", "**", "*metrics.json"), recursive=True)
    
    # Group metric files by directory
    dir_to_files = {}
    for f in metric_files:
        dir_path = os.path.dirname(f)
        if dir_path not in dir_to_files:
            dir_to_files[dir_path] = []
        dir_to_files[dir_path].append(f)
        
    print(f"Found {len(dir_to_files)} evaluation run directories under {base_dir}.\n")
    
    for eval_dir, files in tqdm(sorted(dir_to_files.items()), desc="Gathering all the evaluation results in one .csv file ..."):
        # Extract relative path components for Model and Checkpoint naming
        rel_path = os.path.relpath(eval_dir, base_dir)
        path_parts = rel_path.split(os.sep)
        
        checkpoint_name = os.path.join(*path_parts[1:]) if len(path_parts) > 1 else "default"
        
        cluster_averages, _ = parse_checkpoint_metrics(files)
        
        row = {
            "Model": model_name,
            "Phase": phase,
            "Checkpoint": checkpoint_name,
            "Avg": round(cluster_averages.get("Avg", np.nan), 2),
            "MC STEM": round(cluster_averages.get("MC STEM", np.nan), 2),
            "MC Non-STEM": round(cluster_averages.get("MC Non-STEM", np.nan), 2),
            "GenQA": round(cluster_averages.get("GenQA", np.nan), 2),
            "Math": round(cluster_averages.get("Math", np.nan), 2),
            "Code": round(cluster_averages.get("Code", np.nan), 2),
            "FIM": round(cluster_averages.get("FIM", np.nan), 2),
            "HeldOut": round(cluster_averages.get("HeldOut", np.nan), 2)
        }
        records.append(row)
        
    df = pd.DataFrame(records)
    
    # Reorder and clean columns
    cols = ["Model", "Checkpoint", "Avg", "MC STEM", "MC Non-STEM", "GenQA", "Math", "Code", "FIM", "HeldOut"]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]
    
    # Sort by Model and Checkpoint
    df = df.sort_values(by=["Model", "Checkpoint"]).reset_index(drop=True)
    
    return df


def evaluate(model_name, phase):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Aggregate results
    df_results = aggregate_all_evaluations(BASE_DIR, model_name, phase)
    
    # Save to CSV
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"=== Successfully generated summary table of {model_name} {phase} and saved to: {OUTPUT_CSV} ===\n")
    