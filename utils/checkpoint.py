import os
import json
import shutil


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
        for ckpt in checkpoints:
            shutil.rmtree(os.path.join(output_dir, ckpt), ignore_errors=True)
            print(f"{ckpt} removed")
