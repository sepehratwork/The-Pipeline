import os
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define base paths
BASE_DIR = "/content/drive/MyDrive/Simulated/ModelsCheckpoints"
OUTPUT_DIR = "/content/drive/MyDrive/Simulated"

TABLE2_CSV = os.path.join(OUTPUT_DIR, "table2_detailed_evaluations.csv")
TABLE6_CSV = os.path.join(OUTPUT_DIR, "table6_summary_evaluations.csv")

# Define OLMo 3 Table 2 & Table 43 schema and taxonomy
TABLE2_SCHEMA = [
    ("OlmoBaseEval Math", "Math", [
        ("GSM8k", ["gsm8k"]),
        ("GSM Symbolic", ["gsm_symbolic", "gsm-symbolic"]),
        ("Minerva MATH", ["minerva_math", "minerva-math"]),
        ("MATH", ["math500", "math_500", "math"])
    ]),
    ("OlmoBaseEval Code", "Code", [
        ("BigCodeBench", ["bigcodebench"]),
        ("HumanEval", ["humaneval"]),
        ("DeepSeek LeetCode", ["deepseek_leetcode", "leetcode"]),
        ("DS 1000", ["ds_1000", "ds1000"]),
        ("MBPP", ["mbpp"]),
        ("MultiPL HumanEval", ["multipl_e_humaneval", "multipl_humaneval"]),
        ("MultiPL MBPP", ["multipl_e_mbpp", "multipl_mbpp"])
    ]),
    ("OlmoBaseEval FIM", "FIM", [
        ("HumEval FIM Single", ["fim_single", "humaneval_fim_single"]),
        ("HumEval FIM Random", ["fim_random", "humaneval_fim_random"]),
        ("HumEval FIM Multi", ["fim_multi", "humaneval_fim_multi"])
    ]),
    ("OlmoBaseEval MC STEM", "MC STEM", [
        ("ARC MC", ["arc_mc", "arc_easy", "arc_challenge"]),
        ("MMLU STEM", ["mmlu_stem"]),
        ("MedMCQA MC", ["medmcqa_mc", "medmcqa"]),
        ("MedQA MC", ["medqa_mc", "medqa"]),
        ("SciQ MC", ["sciq_mc", "sciq"])
    ]),
    ("OlmoBaseEval MC Non-STEM", "MC Non-STEM", [
        ("MMLU Humanities", ["mmlu_humanities"]),
        ("MMLU Social Sci.", ["mmlu_social_sci", "mmlu_social_sciences"]),
        ("MMLU Other", ["mmlu_other"]),
        ("CSQA MC", ["csqa_mc", "csqa"]),
        ("PiQA MC", ["piqa_mc", "piqa"]),
        ("SocialIQA MC", ["socialiqa_mc", "socialiqa"]),
        ("CoQA Gen2MC MC", ["coqa_gen2mc", "coqa_mc"]),
        ("DROP Gen2MC MC", ["drop_gen2mc", "drop_mc"]),
        ("Jeopardy Gen2MC MC", ["jeopardy_gen2mc", "jeopardy_mc"]),
        ("NaturalQs Gen2MC MC", ["naturalqs_gen2mc", "naturalqs_mc"]),
        ("SQuAD Gen2MC MC", ["squad_gen2mc", "squad_mc"])
    ]),
    ("OlmoBaseEval GenQA", "GenQA", [
        ("HellaSwag RC", ["hellaswag"]),
        ("Winogrande RC", ["winogrande"]),
        ("Lambada", ["lambada"]),
        ("Basic Skills", ["basic_skills"]),
        ("DROP", ["drop"]),
        ("Jeopardy", ["jeopardy"]),
        ("NaturalQs", ["naturalqs"]),
        ("SQuAD", ["squad"]),
        ("CoQA", ["coqa"])
    ]),
    ("OlmoBaseEval HeldOut", "HeldOut", [
        ("BBH", ["bbh", "bigbench_hard"]),
        ("MMLU Pro MC", ["mmlu_pro"]),
        ("Deepmind Math", ["deepmind_math"]),
        ("LBPP", ["lbpp"])
    ])
]


def get_task_cluster(task_name: str) -> str:
    """Categorizes tasks into OlmoBaseEval clusters matching OLMo 3 paper taxonomy."""
    tn = task_name.lower().replace("-", "_")

    # 1. HeldOut Suite
    if any(k in tn for k in ['mmlu_pro', 'lbpp', 'deepmind_math', 'bbh', 'bigbench_hard']):
        return 'HeldOut'

    # 2. Math Suite
    if any(k in tn for k in ['gsm8k', 'gsm_symbolic', 'minerva_math', 'math500', 'math_500']):
        return 'Math'

    # 3. FIM Suite
    if 'fim' in tn:
        return 'FIM'

    # 4. Code Suite
    if any(k in tn for k in ['humaneval', 'mbpp', 'bigcodebench', 'ds_1000', 'ds1000', 'deepseek_leetcode', 'leetcode', 'multipl_e']):
        return 'Code'

    # Check for Multiple Choice / Gen2MC
    is_mc = ':mc' in tn or '_mc' in tn or 'gen2mc' in tn

    # 5. MC STEM Suite
    stem_keywords = [
        'arc_mc', 'arc_easy', 'arc_challenge', 'medmcqa', 'medqa', 'sciq',
        'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_physics',
        'computer_security', 'conceptual_physics', 'electrical_engineering',
        'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_mathematics',
        'high_school_physics', 'high_school_statistics', 'machine_learning', 'mmlu_stem'
    ]
    if any(k in tn for k in stem_keywords):
        return 'MC STEM'

    # 6. MC Non-STEM Suite
    if is_mc and any(k in tn for k in ['coqa', 'drop', 'jeopardy', 'naturalqs', 'squad', 'csqa', 'piqa', 'socialiqa']):
        return 'MC Non-STEM'

    non_stem_keywords = [
        'csqa', 'piqa', 'socialiqa', 'mmlu_humanities', 'mmlu_social_sci', 'mmlu_other',
        'formal_logic', 'european_history', 'us_history', 'world_history', 'international_law',
        'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
        'prehistory', 'professional_law', 'world_religions', 'econometrics', 'geography',
        'government_and_politics', 'macroeconomics', 'microeconomics', 'psychology',
        'sociology', 'anatomy', 'clinical_knowledge', 'college_medicine', 'global_facts'
    ]
    if any(k in tn for k in non_stem_keywords):
        return 'MC Non-STEM'

    # 7. GenQA Suite (Short-form / RC generative tasks)
    genqa_keywords = ['hellaswag', 'winogrande', 'lambada', 'basic_skills', 'drop', 'jeopardy', 'naturalqs', 'squad', 'coqa']
    if any(k in tn for k in genqa_keywords):
        return 'GenQA'

    if 'mmlu' in tn:
        return 'MC Non-STEM'

    return 'Other'


def parse_checkpoint_metrics(metrics_files):
    """Parses task-*-metrics.json files for a single checkpoint directory."""
    task_scores = {}
    cluster_scores = {}

    for file_path in metrics_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            task_name = data.get('task_name', os.path.basename(file_path).split('-metrics.json')[0])
            metrics = data.get('metrics', {})

            # Extract primary score metric
            score = None
            for key in ['primary_score', 'pass@1', 'acc_raw', 'acc', 'f1', 'exact_match']:
                if key in metrics and metrics[key] is not None:
                    score = metrics[key]
                    break

            if score is None:
                continue

            # Normalize 0.0–1.0 score to 0.0–100.0 percentage scale
            if isinstance(score, (int, float)):
                if 0.0 <= score <= 1.0:
                    score = score * 100.0

            cluster = get_task_cluster(task_name)
            if cluster not in cluster_scores:
                cluster_scores[cluster] = []
            cluster_scores[cluster].append(score)
            task_scores[task_name] = score

        except Exception as e:
            print(f"\n[Warning] Failed to parse file {file_path}: {e}")

    # Compute macro-averages per cluster
    cluster_averages = {cluster: np.mean(scores) for cluster, scores in cluster_scores.items()}

    # Calculate overall Main Suite Avg (6 main clusters in OLMo 3 paper: MC STEM, MC Non-STEM, GenQA, Math, Code, FIM)
    main_clusters = ['MC STEM', 'MC Non-STEM', 'GenQA', 'Math', 'Code', 'FIM']
    avail_main_scores = [cluster_averages[c] for c in main_clusters if c in cluster_averages]

    cluster_averages['Avg'] = np.mean(avail_main_scores) if avail_main_scores else np.nan

    return cluster_averages, task_scores


def find_task_score(task_scores, candidate_keys):
    """Finds best matching score for a benchmark from parsed task scores."""
    # 1. Exact match
    for key in candidate_keys:
        for t_name, score in task_scores.items():
            t_clean = t_name.lower().replace("-", "_")
            if t_clean == key:
                return score

    # 2. Substring match
    for key in candidate_keys:
        for t_name, score in task_scores.items():
            t_clean = t_name.lower().replace("-", "_")
            if key in t_clean:
                # Avoid matching gen2mc variants when searching for standard generative task
                if "gen2mc" in t_clean and "gen2mc" not in key:
                    continue
                if "_mc" in t_clean and "_mc" not in key and "gen2mc" not in key:
                    continue
                return score
    return np.nan


def generate_table2_detailed(checkpoint_data_list):
    """Generates Table 2 format: Detailed benchmark breakdown by cluster."""
    rows = []
    checkpoint_names = [ckpt['Checkpoint'] for ckpt in checkpoint_data_list]

    for cluster_header, cluster_key, task_definitions in TABLE2_SCHEMA:
        # Cluster Header Row (with Macro-Average across checkpoints)
        header_row = {"Benchmark / Capability": f"**{cluster_header}**"}
        for ckpt in checkpoint_data_list:
            avg_score = ckpt['ClusterAverages'].get(cluster_key, np.nan)
            header_row[ckpt['Checkpoint']] = round(avg_score, 1) if not np.isnan(avg_score) else "-"
        rows.append(header_row)

        # Individual Task Rows under this cluster
        for task_label, candidate_keys in task_definitions:
            task_row = {"Benchmark / Capability": f"  {task_label}"}
            for ckpt in checkpoint_data_list:
                score = find_task_score(ckpt['TaskScores'], candidate_keys)
                task_row[ckpt['Checkpoint']] = round(score, 1) if not np.isnan(score) else "-"
            rows.append(task_row)

    df_table2 = pd.DataFrame(rows)
    return df_table2


def generate_table6_summary(checkpoint_data_list):
    """Generates Table 6 format: Summarized cluster macro-averages per checkpoint/mix."""
    summary_rows = []

    for ckpt in checkpoint_data_list:
        c_avgs = ckpt['ClusterAverages']
        row = {
            "Checkpoint": ckpt['Checkpoint'],
            "Avg": round(c_avgs.get("Avg", np.nan), 1),
            "MC STEM": round(c_avgs.get("MC STEM", np.nan), 1),
            "MC Non-STEM": round(c_avgs.get("MC Non-STEM", np.nan), 1),
            "GenQA": round(c_avgs.get("GenQA", np.nan), 1),
            "Math": round(c_avgs.get("Math", np.nan), 1),
            "Code": round(c_avgs.get("Code", np.nan), 1),
            "FIM": round(c_avgs.get("FIM", np.nan), 1),
            "HeldOut": round(c_avgs.get("HeldOut", np.nan), 1)
        }
        summary_rows.append(row)

    df_table6 = pd.DataFrame(summary_rows)
    return df_table6


def evaluate(model_name="olmo3", phase="base"):
    """Main evaluation pipeline with progress bars."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=======================================================")
    print(f"  OLMo 3 Evaluation Pipeline: {model_name.upper()} ({phase})")
    print("=======================================================\n")

    # Step 1: Discover evaluation directories
    search_path = os.path.join(BASE_DIR, model_name, f"{model_name}-{phase}-evaluation-results", "**", "*metrics.json")
    print(f"[1/4] Searching for OLMES metric files in:\n      {search_path}")
    
    metric_files = glob.glob(search_path, recursive=True)
    if not metric_files:
        # Fallback search if path structure differs slightly
        fallback_path = os.path.join(BASE_DIR, "**", "*metrics.json")
        metric_files = glob.glob(fallback_path, recursive=True)

    # Group metric files by checkpoint directory
    dir_to_files = {}
    for f in metric_files:
        dir_path = os.path.dirname(f)
        if dir_path not in dir_to_files:
            dir_to_files[dir_path] = []
        dir_to_files[dir_path].append(f)

    if not dir_to_files:
        print("\n[Error] No metric files found. Please check your BASE_DIR path.")
        return

    print(f"      Found {len(metric_files)} evaluation metric files across {len(dir_to_files)} checkpoints.\n")

    # Step 2: Parse metrics per checkpoint with progress bar
    checkpoint_data_list = []
    
    pbar_parse = tqdm(sorted(dir_to_files.items()), desc="[2/4] Parsing OLMES Checkpoint Results", unit="ckpt")
    for eval_dir, files in pbar_parse:
        rel_path = os.path.relpath(eval_dir, BASE_DIR)
        path_parts = rel_path.split(os.sep)
        checkpoint_name = os.path.join(*path_parts[1:]) if len(path_parts) > 1 else "Checkpoint"
        
        pbar_parse.set_postfix({"current": checkpoint_name[:20]})

        cluster_averages, task_scores = parse_checkpoint_metrics(files)
        
        checkpoint_data_list.append({
            "Checkpoint": checkpoint_name,
            "ClusterAverages": cluster_averages,
            "TaskScores": task_scores
        })

    # Step 3: Generate Table 2 and Table 6
    pbar_gen = tqdm(total=2, desc="[3/4] Building Table 2 (Detailed) & Table 6 (Summary)", unit="table")
    
    df_table2 = generate_table2_detailed(checkpoint_data_list)
    pbar_gen.update(1)
    
    df_table6 = generate_table6_summary(checkpoint_data_list)
    pbar_gen.update(1)
    pbar_gen.close()

    # Step 4: Export to CSV and display formatted outputs
    print("\n[4/4] Saving generated tables to disk...")
    
    df_table2.to_csv(TABLE2_CSV, index=False)
    print(f"  ✔ Table 2 (Detailed) saved to: {TABLE2_CSV}")
    
    df_table6.to_csv(TABLE6_CSV, index=False)
    print(f"  ✔ Table 6 (Summary)  saved to: {TABLE6_CSV}")

    # Display Tables in terminal
    print("\n" + "="*80)
    print("  TABLE 6: OLMo 3 Base Model Summary Evaluation Suite")
    print("="*80)
    print(df_table6.to_string(index=False))

    print("\n" + "="*80)
    print("  TABLE 2: OLMo 3 Detailed Task-Level Evaluation Breakdown")
    print("="*80)
    print(df_table2.to_string(index=False))
    print("\nEvaluation pipeline complete!\n")


if __name__ == "__main__":
    evaluate(model_name="olmo3", phase="base")