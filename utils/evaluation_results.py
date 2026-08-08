import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

# Structure matching Table 2 & Table 43 from OLMo 3 Paper
TASK_TAXONOMY = {
    "OlmoBaseEval Math": {
        "gsm8k": ["gsm8k"],
        "gsm_symbolic": ["gsm_symbolic"],
        "math": ["minerva_math", "math500", "math_500", "math"]
    },
    "OlmoBaseEval Code": {
        "bigcodebench": ["bigcodebench"],
        "humaneval": ["humaneval"],
        "deepseek_leetcode": ["deepseek_leetcode", "leetcode"],
        "ds_1000": ["ds_1000", "ds1000"],
        "mbpp": ["mbpp"],
        "multipl_humaneval": ["multipl_e_humaneval", "multipl_humaneval"],
        "multipl_mbpp": ["multipl_e_mbpp", "multipl_mbpp"]
    },
    "OlmoBaseEval FIM": {
        "humeval_fim_single": ["humeval_fim_single"],
        "humeval_fim_random": ["humeval_fim_random"],
        "humeval_fim_multi": ["humeval_fim_multi"]
    },
    "OlmoBaseEval MC STEM": {
        "arc_mc": ["arc_mc", "arc_easy", "arc_challenge"],
        "mmlu_stem": ["mmlu_stem"],
        "medmcqa_mc": ["medmcqa_mc", "medmcqa"],
        "medqa_mc": ["medqa_mc", "medqa"],
        "sciq_mc": ["sciq_mc", "sciq"]
    },
    "OlmoBaseEval MC Non-STEM": {
        "mmlu_humanities": ["mmlu_humanities"],
        "mmlu_social_sci": ["mmlu_social_sci", "mmlu_social_science"],
        "mmlu_other": ["mmlu_other"],
        "csqa_mc": ["csqa_mc", "csqa"],
        "piqa_mc": ["piqa_mc", "piqa"],
        "socialiqa_mc": ["socialiqa_mc", "socialiqa"],
        "coqa_gen2mc_mc": ["coqa_gen2mc", "coqa_mc"],
        "drop_gen2mc_mc": ["drop_gen2mc", "drop_mc"],
        "jeopardy_gen2mc_mc": ["jeopardy_gen2mc", "jeopardy_mc"],
        "naturalqs_gen2mc_mc": ["naturalqs_gen2mc", "naturalqs_mc"],
        "squad_gen2mc_mc": ["squad_gen2mc", "squad_mc"]
    },
    "OlmoBaseEval GenQA": {
        "hellaswag_rc": ["hellaswag", "hellaswag_rc"],
        "winogrande_rc": ["winogrande", "winogrande_rc"],
        "lambada": ["lambada"],
        "basic_skills": ["basic_skills"],
        "drop": ["drop"],
        "jeopardy": ["jeopardy"],
        "naturalqs": ["naturalqs"],
        "squad": ["squad"],
        "coqa": ["coqa"]
    },
    "OlmoBaseEval HeldOut": {
        "bbh": ["bbh", "bigbench_hard"],
        "mmlu_pro_mc": ["mmlu_pro", "mmlu_pro_mc"],
        "deepmind_math": ["deepmind_math"],
        "lbpp": ["lbpp"]
    }
}


def map_task_to_cluster_and_name(raw_task_name: str) -> Tuple[str, str]:
    """
    Maps a raw OLMES task name to its cluster and standard benchmark name.
    """
    tn = raw_task_name.lower().strip()
    
    for cluster_name, tasks in TASK_TAXONOMY.items():
        for std_task_name, aliases in tasks.items():
            for alias in aliases:
                if alias in tn:
                    # Distinguish between Gen2MC vs GenQA variants
                    if "gen2mc" in alias or "gen2mc" in tn:
                        if cluster_name == "OlmoBaseEval MC Non-STEM":
                            return cluster_name, std_task_name
                    elif cluster_name == "OlmoBaseEval GenQA" and "gen2mc" in tn:
                        continue
                    else:
                        return cluster_name, std_task_name
                        
    return "Other", raw_task_name


def parse_checkpoint_metrics(metrics_files: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Parses task-*-metrics.json files for a single evaluation run directory.
    Returns:
        task_scores: dict mapping standardized task names to scores.
        cluster_averages: dict mapping cluster names to macro-averaged scores.
    """
    task_scores = {}
    cluster_scores = {cluster: {} for cluster in TASK_TAXONOMY.keys()}
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            raw_task_name = data.get('task_name', os.path.basename(file_path).rsplit('-metrics.json', 1)[0])
            metrics = data.get('metrics', {})
            
            # Extract primary metric score
            score = metrics.get('primary_score', metrics.get('acc_raw', metrics.get('pass@1', metrics.get('f1', None))))
            if score is None:
                continue
                
            # Convert 0.0-1.0 scale to percentage 0.0-100.0
            if isinstance(score, (int, float)):
                if 0.0 <= score <= 1.0:
                    score = score * 100.0
            
            cluster, std_task_name = map_task_to_cluster_and_name(raw_task_name)
            
            task_scores[std_task_name] = score
            if cluster in cluster_scores:
                cluster_scores[cluster][std_task_name] = score
                
        except Exception as e:
            tqdm.write(f"⚠️ Warning: Failed to parse {file_path}: {e}")
            
    # Calculate macro-averages per cluster
    cluster_averages = {}
    for cluster_name, tasks_dict in cluster_scores.items():
        if tasks_dict:
            cluster_averages[cluster_name] = float(np.mean(list(tasks_dict.values())))
        else:
            cluster_averages[cluster_name] = np.nan
            
    # Calculate Main Suite Overall Average (MC STEM, MC Non-STEM, GenQA, Math, Code, FIM)
    main_clusters = ["OlmoBaseEval MC STEM", "OlmoBaseEval MC Non-STEM", "OlmoBaseEval GenQA", "OlmoBaseEval Math", "OlmoBaseEval Code", "OlmoBaseEval FIM"]
    main_scores = [cluster_averages[c] for c in main_clusters if c in cluster_averages and not np.isnan(cluster_averages[c])]
    
    if main_scores:
        cluster_averages["Avg"] = float(np.mean(main_scores))
    else:
        cluster_averages["Avg"] = np.nan
        
    return cluster_averages, task_scores


def collect_evaluation_data(base_dir: str, model_name: str, phase: str) -> Dict[str, Dict[str, Any]]:
    """
    Scans the base directory and gathers metrics for all checkpoints.
    """
    search_pattern = os.path.join(base_dir, model_name, f"{model_name}-{phase}-evaluation-results", "**", "*metrics.json")
    
    print(f"🔍 [1/4] Searching for evaluation metric files in: {search_pattern}")
    metric_files = glob.glob(search_pattern, recursive=True)
    
    if not metric_files:
        # Fallback search if path structure differs slightly
        fallback_pattern = os.path.join(base_dir, "**", "*metrics.json")
        print(f"🔄 Retrying with fallback search pattern: {fallback_pattern}")
        metric_files = glob.glob(fallback_pattern, recursive=True)
        
    # Group metric files by run directory
    dir_to_files = {}
    for f in metric_files:
        dir_path = os.path.dirname(f)
        dir_to_files.setdefault(dir_path, []).append(f)
        
    print(f"✓ Found {len(metric_files)} metric files across {len(dir_to_files)} evaluation checkpoints.\n")
    
    results = {}
    pbar = tqdm(sorted(dir_to_files.items()), desc="📊 [2/4] Parsing Checkpoint Metrics", unit="ckpt")
    
    for eval_dir, files in pbar:
        rel_path = os.path.relpath(eval_dir, base_dir)
        path_parts = rel_path.split(os.sep)
        checkpoint_name = os.path.join(*path_parts[1:]) if len(path_parts) > 1 else os.path.basename(eval_dir)
        pbar.set_postfix({"checkpoint": checkpoint_name[:25]})
        
        cluster_averages, task_scores = parse_checkpoint_metrics(files)
        
        results[checkpoint_name] = {
            "cluster_averages": cluster_averages,
            "task_scores": task_scores
        }
        
    return results


def build_table6_summary(parsed_data: Dict[str, Dict[str, Any]], model_name: str) -> pd.DataFrame:
    """
    Generates Table 6 style summary (Rows = Checkpoints, Columns = Macro Cluster Averages).
    """
    print("\n📝 [3/4] Constructing Table 6 (Macro Summary)...")
    records = []
    
    for ckpt_name, data in tqdm(parsed_data.items(), desc="   Building Summary Table"):
        avgs = data["cluster_averages"]
        
        row = {
            "Model": model_name,
            "Checkpoint": ckpt_name,
            "Avg": round(avgs.get("Avg", np.nan), 1),
            "MC STEM": round(avgs.get("OlmoBaseEval MC STEM", np.nan), 1),
            "MC Non-STEM": round(avgs.get("OlmoBaseEval MC Non-STEM", np.nan), 1),
            "GenQA": round(avgs.get("OlmoBaseEval GenQA", np.nan), 1),
            "Math": round(avgs.get("OlmoBaseEval Math", np.nan), 1),
            "Code": round(avgs.get("OlmoBaseEval Code", np.nan), 1),
            "FIM": round(avgs.get("OlmoBaseEval FIM", np.nan), 1),
            "HeldOut": round(avgs.get("OlmoBaseEval HeldOut", np.nan), 1)
        }
        records.append(row)
        
    df_summary = pd.DataFrame(records)
    return df_summary.sort_values(by="Checkpoint").reset_index(drop=True)


def build_table2_detailed(parsed_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Generates Table 2 style detailed breakdown (Rows = Tasks/Clusters, Columns = Checkpoints).
    """
    print("📝 [4/4] Constructing Table 2 (Detailed Benchmark Breakdown)...")
    checkpoints = sorted(list(parsed_data.keys()))
    
    table_rows = []
    
    pbar = tqdm(TASK_TAXONOMY.items(), desc="   Building Detailed Table")
    for cluster_name, tasks_dict in pbar:
        pbar.set_postfix({"Cluster": cluster_name.replace("OlmoBaseEval ", "")})
        
        # 1. Cluster Header Row (Macro-Average)
        cluster_row = {"Category / Benchmark": f"=== {cluster_name} ==="}
        for ckpt in checkpoints:
            avg_val = parsed_data[ckpt]["cluster_averages"].get(cluster_name, np.nan)
            cluster_row[ckpt] = round(avg_val, 1) if not np.isnan(avg_val) else "-"
        table_rows.append(cluster_row)
        
        # 2. Individual Benchmark Rows under the cluster
        for std_task_name in tasks_dict.keys():
            task_row = {"Category / Benchmark": f"  - {std_task_name}"}
            for ckpt in checkpoints:
                score = parsed_data[ckpt]["task_scores"].get(std_task_name, np.nan)
                task_row[ckpt] = round(score, 1) if not np.isnan(score) else "-"
            table_rows.append(task_row)
            
    df_detailed = pd.DataFrame(table_rows)
    return df_detailed


def evaluate(model_name: str, phase: str, base_dir: str, output_summary_csv: str = None, output_detailed_csv: str = None):
    """
    Main evaluation pipeline.
    """
    print("=" * 80)
    print(f"🚀 Starting OLMo 3 Evaluation Processing for [{model_name.upper()} - {phase.upper()}]")
    print("=" * 80 + "\n")
    
    # Define default output paths if not specified
    if output_summary_csv is None:
        output_summary_csv = os.path.join(base_dir, f"{model_name}_{phase}_summary_table6.csv")
    if output_detailed_csv is None:
        output_detailed_csv = os.path.join(base_dir, f"{model_name}_{phase}_detailed_table2.csv")
        
    # 1. Collect Data
    parsed_data = collect_evaluation_data(base_dir, model_name, phase)
    
    if not parsed_data:
        print("❌ No evaluation metric files found. Please check base_dir and folder structure.")
        return
        
    # 2. Build Summary Table (Table 6)
    df_summary = build_table6_summary(parsed_data, model_name)
    
    # 3. Build Detailed Table (Table 2)
    df_detailed = build_table2_detailed(parsed_data)
    
    # 4. Export CSVs
    os.makedirs(os.path.dirname(os.path.abspath(output_summary_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_detailed_csv)), exist_ok=True)
    
    df_summary.to_csv(output_summary_csv, index=False)
    df_detailed.to_csv(output_detailed_csv, index=False)
    
    # 5. Display Formatted Output
    print("\n" + "=" * 80)
    print("📋 SUMMARY TABLE (Table 6 Style - Macro Averages)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print(f"\n💾 Saved Summary Table to: {output_summary_csv}\n")
    
    print("=" * 80)
    print("🔬 DETAILED TABLE (Table 2 Style - Per-Benchmark Hierarchy)")
    print("=" * 80)
    print(df_detailed.to_string(index=False))
    print(f"\n💾 Saved Detailed Table to: {output_detailed_csv}\n")
    print("✅ All evaluation tables successfully generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and generate OLMo 3 paper style evaluation tables (Table 2 and Table 6) from OLMES results."
    )
    
    parser.add_argument(
        "--model-name", "-m",
        type=str,
        default="olmo3",
        help="Name of the model directory/family (e.g., olmo3, my_custom_model)"
    )
    parser.add_argument(
        "--phase", "-p",
        type=str,
        default="base",
        help="Training phase (e.g., base, think, instruct, rlzero)"
    )
    parser.add_argument(
        "--base-dir", "-b",
        type=str,
        default="/content/drive/MyDrive/Simulated/ModelsCheckpoints",
        help="Path to base checkpoints directory"
    )
    parser.add_argument(
        "--output-summary-csv", "-s",
        type=str,
        default=None,
        help="Custom output file path for summary CSV (Table 6 style)"
    )
    parser.add_argument(
        "--output-detailed-csv", "-d",
        type=str,
        default=None,
        help="Custom output file path for detailed CSV (Table 2 style)"
    )

    args = parser.parse_args()

    evaluate(
        model_name=args.model_name,
        phase=args.phase,
        base_dir=args.base_dir,
        output_summary_csv=args.output_summary_csv,
        output_detailed_csv=args.output_detailed_csv
    )