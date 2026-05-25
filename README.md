---

# 🚀 LLM-Forge: End-to-End 6-Stage LLM Training Framework

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**LLM-Forge** is a highly modular, memory-efficient, and extensible framework designed to take Large Language Models (LLMs) from raw initialization all the way to advanced reasoning alignment. 

Inspired by modern scaling laws and cutting-edge alignment techniques (like DeepSeek-R1), this framework implements a rigorous **6-Stage Training Pipeline**, culminating in **Reinforcement Learning with Verifiable Rewards (RLVR)**.

---

## ✨ Key Features

*   **End-to-End 6-Stage Pipeline**: Seamlessly transition from Pre-training ➡️ Mid-training ➡️ Long-Context Extension ➡️ SFT ➡️ DPO ➡️ RLVR (GRPO).
*   **Plug-and-Play Architecture**: Easily swap out language models (OLMo 3, DeepSeek, Qwen) and RL algorithms (GRPO, PPO, DAPO) without rewriting training loops.
*   **Advanced Memory Management**: Built-in support for `bfloat16`, Gradient Checkpointing, Grouped Query Attention (GQA), and custom KV-cache generation loops to prevent OOM errors.
*   **Real-time Training Analytics**: Custom callbacks automatically track and plot Gradient Variance, Gradient Entropy, Training Loss, and Cumulative FLOPs.
*   **Verifiable Reasoning (RLVR)**: Stage 6 includes a custom generation loop that rewards models for explicit reasoning chains (e.g., `<think>...</think>`) and exact-match ground truth accuracy.

---

## 📂 Architecture & Directory Structure

The codebase is strictly organized by separation of concerns, making it trivial to scale and maintain.

```text
LLM-Forge/
├── models/                 # Model architectures & configurations
│   ├── __init__.py         # Model registry (add new models here)
│   └── olmo3.py            # OLMo 3 implementation
├── rl_algorithms/          # Reinforcement Learning algorithms
│   ├── __init__.py         # RL registry
│   ├── base.py             # Abstract base class for RL algorithms
│   └── grpo.py             # Group Relative Policy Optimization
├── data/                   # Dataset loading and formatting utilities
│   ├── __init__.py
│   └── dataset_utils.py    # SFT, DPO, and Pre-training formatters
├── utils/                  # Shared utilities
│   ├── __init__.py
│   ├── callbacks.py        # Gradient & FLOPs tracking
│   ├── checkpoint.py       # Resumption logic
│   └── generation.py       # Safe KV-cache generation loop
├── pipelines/              # Training stage logic
│   ├── __init__.py
│   ├── pretrain.py         # Stages 1, 2, and 3
│   ├── posttrain.py        # Stages 4 and 5
│   └── rlvr.py             # Stage 6
├── main.py                 # Main entry point
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/LLM-Forge.git
   cd LLM-Forge
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure your environment supports CUDA if you plan to train on GPUs).*

---

## 🚀 Quick Start

To run the entire 6-stage pipeline sequentially, simply execute the main script. 

```bash
python main.py
```

*Note: By default, `main.py` is configured to run a scaled-down version of OLMo 3 (~500M parameters) for demonstration purposes. You can adjust the hyperparameters in the respective pipeline files.*

---

## 🗺️ The 6-Stage Pipeline Explained

### Phase 1: Pre-Training (`pipelines/pretrain.py`)
*   **Stage 1: Pre-training**: Standard causal language modeling on massive web corpora (e.g., Dolma).
*   **Stage 2: Mid-training**: Annealing phase using high-quality, domain-specific data mixes.
*   **Stage 3: Long-context Extension**: Expands the context window (e.g., 2k ➡️ 4k/8k) using YaRN (Yet another RoPE extensioN).

### Phase 2: Post-Training (`pipelines/posttrain.py`)
*   **Stage 4: Supervised Fine-Tuning (SFT)**: Teaches the model chat formatting and basic instruction following.
*   **Stage 5: Direct Preference Optimization (DPO)**: Aligns the model with human preferences using chosen/rejected response pairs.

### Phase 3: Reasoning Alignment (`pipelines/rlvr.py`)
*   **Stage 6: RLVR (GRPO)**: Reinforcement Learning with Verifiable Rewards. The model generates multiple completions, and rewards are calculated based on formatting (e.g., using `<think>` tags) and factual accuracy. The policy is updated using Group Relative Policy Optimization (GRPO).

---

## 📊 Monitoring & Logging

The framework automatically generates a `training_metrics.png` and a `training_log.jsonl` in the output directory of every stage. It tracks:
1. **Gradient Variance**: To monitor training stability.
2. **Gradient Entropy**: To measure the distribution of learning across parameters.
3. **Training Loss**: Standard cross-entropy + Z-loss.
4. **Cumulative FLOPs**: To measure computational efficiency.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.