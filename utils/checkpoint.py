import os
import json
import shutil
from huggingface_hub import HfApi


def save_to_hf_hub(model_path, tokenizer, repo_name, hf_username=None):
    """
    Saves the trained model and tokenizer to Hugging Face Hub if it hasn't been uploaded yet.
    
    Args:
        model_path (str): Path to the saved final model directory.
        tokenizer: PreTrainedTokenizer instance.
        repo_name (str): Target repository name on Hugging Face (e.g., 'olmo3_base').
        hf_username (str, optional): Hugging Face username or organization. If None, auto-detected.
    """
    api = HfApi()
    
    # Determine repo_id (e.g., 'username/olmo3_base')
    if hf_username:
        repo_id = f"{hf_username}/{repo_name}"
    else:
        try:
            user_info = api.whoami()
            username = user_info.get("name")
            repo_id = f"{username}/{repo_name}" if username else repo_name
        except Exception:
            repo_id = repo_name

    # Check whether the model repository already exists on Hugging Face Hub
    try:
        exists = api.repo_exists(repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"Warning: Could not check existence of '{repo_id}' on HF Hub: {e}")
        exists = False

    if exists:
        print(f"Model '{repo_id}' already exists on Hugging Face Hub. Skipping upload.")
        return

    print(f"Uploading model from '{model_path}' to Hugging Face Hub as '{repo_id}'...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model"
        )
        if tokenizer is not None:
            tokenizer.push_to_hub(repo_id)
        print(f"Successfully uploaded '{repo_id}' to Hugging Face Hub.")
    except Exception as e:
        print(f"Failed to upload model to Hugging Face Hub: {e}")


def get_latest_checkpoint(output_dir):
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            return os.path.join(output_dir, checkpoints[-1])
    return None


def get_resume_state(log_file):
    last_step = -1
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    last_step = data.get('step', last_step)
            f.close()
    return last_step


def cleanup_checkpoints(output_dir, keep=2):
    """Keep only the last `keep` checkpoints during training."""
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if len(checkpoints) > keep:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            # Remove all but the last `keep` checkpoints
            for ckpt in checkpoints[:-keep]:
                shutil.rmtree(os.path.join(output_dir, ckpt), ignore_errors=True)


def clear_all_checkpoints(output_dir):
    """Remove all checkpoints after the phase is completely finished."""
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        print(f"Checkpoints: {checkpoints}")
        for ckpt in checkpoints:
            shutil.rmtree(os.path.join(output_dir, ckpt), ignore_errors=True)
            print(f"{ckpt} removed")