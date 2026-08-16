import os
import json
import shutil
from huggingface_hub import HfApi


def save_to_hf_hub(model_path, repo_name, hf_username=None):
    """
    Saves the trained model to Hugging Face Hub if it hasn't been uploaded yet.
    
    Args:
        model_path (str): Path to the saved final model directory.
        repo_name (str): Target repository name on Hugging Face (e.g., 'olmo3_base').
        hf_username (str, optional): Hugging Face username or organization. If None, auto-detected.
    """
    api = HfApi()
    
    if hf_username:
        repo_id = f"{hf_username}/{repo_name}"
    else:
        raise ValueError("hf_username must have value")

    # # Check whether the model repository already exists on Hugging Face Hub
    # try:
    #     exists = api.repo_exists(repo_id=repo_id, repo_type="model")
    # except Exception as e:
    #     print(f"Warning: Could not check existence of '{repo_id}' on HF Hub: {e}")
    #     exists = False

    # if exists:
    #     print(f"Model '{repo_id}' already exists on Hugging Face Hub. Skipping upload.")
    #     return

    print(f"📤 Uploading folder '{model_path}' to Hugging Face Hub repo '{repo_id}'...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✓ Successfully published '{repo_id}' on Hugging Face Hub!")
    except Exception as e:
        print(f"❌ Failed to upload model to Hugging Face Hub: {e}")


def get_latest_checkpoint(output_dir):
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest = os.path.join(output_dir, checkpoints[-1])
            print(f"🔍 Located latest checkpoint: {latest}")
            return latest
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
    print(f"ℹ️  Resume state loaded: last recorded step = {last_step}")
    return last_step


def cleanup_checkpoints(output_dir, keep=2):
    """Keep only the last `keep` checkpoints during training."""
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if len(checkpoints) > keep:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            # Remove all but the last `keep` checkpoints
            for ckpt in checkpoints[:-keep]:
                target = os.path.join(output_dir, ckpt)
                shutil.rmtree(target, ignore_errors=True)
                print(f"🧹 Pruned older checkpoint: {ckpt}")


def clear_all_checkpoints(output_dir):
    """Remove all checkpoints after the phase is completely finished."""
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            print(f"🧹 Cleaning up {len(checkpoints)} intermediate checkpoint(s) in {output_dir}...")
            for ckpt in checkpoints:
                shutil.rmtree(os.path.join(output_dir, ckpt), ignore_errors=True)
                print(f"  └ Removed: {ckpt}")
            print("✓ Checkpoint directory cleanup complete.")