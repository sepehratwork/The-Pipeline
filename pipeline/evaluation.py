import os
import sys
import json
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_subprocess(cmd, env=None):
    """
    Helper function to execute shell commands and print logs in real-time.
    """
    logger.info(f"Running evaluation command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            logger.error(f"OLMES Command failed with exit code {process.returncode}")
            return False
        logger.info("OLMES Command executed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during OLMES command execution: {e}")
        return False

def run_olmes_evaluation(
    model_path: str,
    output_dir: str,
    stage: str,
    use_vllm: bool = True,
    openai_api_key: str = None,
    hf_token: str = None,
    extra_args: list = None
):
    """
    Launches standard OLMES evaluation configurations matching the current model stage.
    
    Args:
        model_path (str): Path to the saved Hugging Face model checkpoint directory or Hub ID.
        output_dir (str): Directory where the OLMES metrics and outputs will be written.
        stage (str): The current training stage ('pretraining', 'midtraining', 'long_context', 
                     'sft', 'dpo', or 'rlvr').
        use_vllm (bool): Flag to utilize vLLM backend for faster inference during evaluation.
        openai_api_key (str): OpenAI API key required for LLM-as-a-judge (e.g. SimpleQA, AlpacaEval).
        hf_token (str): Hugging Face authentication token for safety models or gated datasets.
        extra_args (list): Additional CLI arguments to append to the command.
    """
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"Model path {model_path} does not exist. Skipping evaluation.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    
    env = os.environ.copy()
    if openai_api_key:
        env["OPENAI_API_KEY"] = openai_api_key
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HF_AUTH_TOKEN"] = hf_token

    logger.info(f"Starting OLMES evaluation process for stage '{stage}' at path: {model_path}")
    
    is_base_model = stage in ["pretraining", "midtraining", "long_context"]
    is_instruct_model = stage in ["sft", "dpo", "rlvr"]
    success = True

    if is_base_model:
        # standard base evaluation benchmarks for OLMo 3
        tasks = [
            # Base Easy Suite (small-scale proxy metrics)
            "olmo3:base_easy:code_bpb",
            "olmo3:base_easy:math_bpb",
            "olmo3:base_easy:qa_rc",
            "olmo3:base_easy:qa_bpb",
            # Base Main Suite (full evaluation)
            "olmo3:base:stem_qa_mc",
            "olmo3:base:nonstem_qa_mc",
            "olmo3:base:gen",
            "olmo3:base:math",
            "olmo3:base:code",
            "olmo3:base:code_fim",
            # Base Held-out Suite
            "olmo3:heldout"
        ]

        cmd = [
            "olmes",
            "--model", model_path,
            "--output-dir", output_dir,
            "--task"
        ] + tasks

        if use_vllm:
            cmd += ["--model-type", "vllm"]
            
        if extra_args:
            cmd += extra_args

        success = success and run_subprocess(cmd, env=env)

    elif is_instruct_model:
        # standard post-training evaluation benchmarks for OLMo 3 instruct models
        tasks_adapt = ["olmo3:adapt"]
        cmd_adapt = [
            "olmes",
            "--model", model_path,
            "--output-dir", output_dir,
            "--task"
        ] + tasks_adapt

        if use_vllm:
            cmd_adapt += ["--model-type", "vllm"]
            
        if extra_args:
            cmd_adapt += extra_args

        success = success and run_subprocess(cmd_adapt, env=env)

        # Safety evaluation using hf-safety-eval as the reference model
        max_length = 32768 if stage == "rlvr" else 2048
        max_gen_toks = 32768 if stage == "rlvr" else 2048

        task_args = {
            "generation_kwargs": {
                "max_gen_toks": max_gen_toks,
                "truncate_context": False
            }
        }

        model_args = {
            "model_path": model_path,
            "max_length": max_length,
            "trust_remote_code": True
        }
        if stage == "rlvr":
            model_args["process_output"] = "r1_style"

        cmd_safety = [
            "oe-eval",
            "--model", "hf-safety-eval",
            "--task", "safety::olmo3",
            "--task-args", json.dumps(task_args),
            "--model-args", json.dumps(model_args),
            "--output-dir", output_dir
        ]
        
        if extra_args:
            cmd_safety += extra_args

        success = success and run_subprocess(cmd_safety, env=env)

    else:
        logger.error(f"Unrecognized OLMo 3 stage: '{stage}'. Unable to execute OLMES tasks.")
        return False

    if success:
        logger.info(f"OLMES evaluation for {model_path} successfully completed.")
    else:
        logger.warning(f"One or more OLMES evaluations failed for model: {model_path}.")

    return success