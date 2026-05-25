import os
import json

def get_latest_checkpoint(output_dir):
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            return os.path.join(output_dir, checkpoints[-1])
    return None

def get_resume_state(log_file):
    last_step = 0
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    last_step = data.get('step', last_step)
    return last_step
