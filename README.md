# LLM Evaluation Pipeline for CS-PROB Dataset

Automated evaluation pipeline for testing large language models on computer science problems across multiple domains (Networking, ML, Database, Algorithms & Data Structures, Software Engineering).

## 📋 Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Setup Instructions](#setup-instructions)
- [Dataset Structure](#dataset-structure)
- [Running Evaluations](#running-evaluations)
- [Model Configuration](#model-configuration)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

This pipeline evaluates multiple LLMs on a curated dataset of computer science questions, using an LLM-as-a-judge approach for automated scoring.

**Current Status:**
- ✅ Dataset preprocessing with domain mapping
- ✅ Multi-model evaluation support
- ✅ 4-bit quantization for memory efficiency
- ✅ Automatic checkpointing every 100 queries
- ✅ Append mode for incremental model evaluation
- 🔄 **Dataset**: 930 questions across 5 domains
- 🔄 **Evaluation Models**: Llama-3-8B, Mistral-7B-v0.1
- 🔄 **Judge Model**: Qwen2.5-72B-Instruct

## 💻 System Requirements

### Minimum Requirements
- **GPU**: 24GB VRAM (e.g., RTX 3090, RTX 4090, A5000)
- **RAM**: 32GB system memory
- **Storage**: 100GB free space (for models and results)
- **CUDA**: Version 11.8 or higher
- **Python**: 3.10+

### Recommended Requirements (Current Setup)
- **GPU**: 48GB+ VRAM (e.g., RTX 6000 Ada, A6000, dual GPUs)
- **RAM**: 64GB+ system memory
- **Storage**: 200GB+ free space
- **Multi-GPU**: Automatic distribution via `device_map="auto"`

### Tested Configuration
- 2x NVIDIA RTX 6000 Ada (48GB each, 96GB total VRAM)
- 754GB RAM
- Ubuntu Linux with CUDA 12.x

##  Setup Instructions

### 1. Clone Repository
```bash
git clone <repository-url>
cd intro-llm-project
```

### 2. Create Conda Environment
```bash
conda create -n llm-judge python=3.10 -y
conda activate llm-judge
```

### 3. Install Dependencies

**Option A: Using requirements.txt (Recommended)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Option B: Manual Installation**
```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.3 accelerate==1.12.0 bitsandbytes==0.48.2
pip install pandas openpyxl tqdm python-dotenv
```

**Tested Package Versions:**
- `torch`: 2.9.1 (CUDA 12.1)
- `transformers`: 4.57.3
- `accelerate`: 1.12.0
- `bitsandbytes`: 0.48.2
- `pandas`: 2.0+
- `python`: 3.10

### 4. Configure Hugging Face Token
Create a `.env` file in the project root:
```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

Get your token from: https://huggingface.co/settings/tokens

### 5. Prepare Dataset
Place `CS-PROB Dataset.xlsx` in the `data/` folder, then run:
```bash
python scripts/preprocess_dataset.py
```

This will:
- Extract questions from specific sheets (Tasmia done, Utshab, Sazedur, Nawfal, Nieb)
- Assign Q_ID to each question (starting from 1)
- Map sheets to domains (Networking, ML, Database, Algo & DS, SWE)
- Save to `data/processed_csprob.csv`

### 6. Configure Model Cache (Optional)
By default, models download to Hugging Face cache. To use a custom location:
```python
# In scripts/evaluate.py, line 59
cache_dir = "/your/custom/path/models"  # Change this path
```

## 📊 Dataset Structure

### Input: `data/processed_csprob.csv`
| Column | Description |
|--------|-------------|
| Q_ID | Unique question identifier (1-930) |
| Domain | Question category (Networking, ML, Database, etc.) |
| Question | The question text |
| Answer | Reference answer |
| Difficulty | Easy, Medium, Hard |
| Source | Source reference |

### Domain Distribution
- **Networking**: 202 questions
- **Software Engineering**: 207 questions
- **Machine Learning**: 200 questions
- **Algorithms & Data Structures**: 200 questions
- **Database**: 121 questions
- **Total**: 930 questions

## 🎯 Running Evaluations

### Test Mode (2 questions)
```bash
python scripts/evaluate.py --test
```

### Full Evaluation
```bash
python scripts/evaluate.py
```

### Add New Model (Append Mode)
1. Edit `scripts/evaluate.py` and add model to `MODEL_CONFIGS`:
```python
MODEL_CONFIGS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "qwen-7b": "Qwen/Qwen-7B",  # Add new model
}
```

2. Run with `--append` flag:
```bash
python scripts/evaluate.py --append
```

This will add new model columns without re-evaluating existing models.

### Monitor Progress
- **Terminal**: Progress bar shows current question and completion percentage
- **Log File**: `evaluation/evaluation.log` contains detailed logs
- **Checkpoints**: Auto-saved every 100 queries to `evaluation/results_checkpoint_*.csv`

## ⚙️ Model Configuration

### Current Models

| Model | Parameters | Quantization | VRAM Usage | Purpose |
|-------|-----------|--------------|------------|---------|
| Meta-Llama-3-8B | 8B | 4-bit | ~5GB | Evaluation |
| Mistral-7B-v0.1 | 7B | 4-bit | ~4GB | Evaluation |
| Qwen2.5-72B-Instruct | 72B | 4-bit | ~40GB | Judge |

### Quantization Options

Edit `scripts/evaluate.py` to change quantization levels:

```python
# Line 164-166: Evaluation models
models = {name: load_model(mid, TORCH_DEVICE, quantization_bits=4) for name, mid in MODEL_CONFIGS.items()}

# Line 168: Judge model
judge_models = {name: load_model(path, TORCH_DEVICE, quantization_bits=4) for name, path in JUDGE_MODELS.items()}
```

**Available Options:**
- `quantization_bits=4`: 4-bit (lowest VRAM, faster, slight quality loss)
- `quantization_bits=8`: 8-bit (medium VRAM, balanced)
- `quantization_bits=None`: FP16 (highest VRAM, best quality)

### Performance Tuning

**Batch Size** (Line 50):
```python
BATCH_SIZE = 32  # Adjust based on available RAM
```
- Higher = faster but more RAM usage
- Recommended: 16-32 for 64GB+ RAM, 8-16 for 32GB RAM

**Token Limits** (Lines 114, 135):
```python
max_new_tokens=512  # For evaluation models
max_new_tokens=256  # For judge model
```
- Lower = faster inference but shorter responses
- Recommended: 256-512 for evaluation, 128-256 for judge

## 📈 Output Format

### Results: `evaluation/results.csv`

| Column | Description |
|--------|-------------|
| Q_ID | Question identifier |
| Domain | Question domain |
| Difficulty | Question difficulty level |
| Question | Question text |
| Reference_Answer | Ground truth answer |
| Meta-Llama-3-8B(4bit)_Response | Llama-3 model response |
| Meta-Llama-3-8B(4bit)_Score | Llama-3 score (0-1) |
| Mistral-7B-v0.1(4bit)_Response | Mistral model response |
| Mistral-7B-v0.1(4bit)_Score | Mistral score (0-1) |
| ... | Additional models as columns |

**Each model evaluation adds two columns:**
1. `{ModelName}(4bit)_Response`: The model's generated answer
2. `{ModelName}(4bit)_Score`: Judge's score (0.0 to 1.0)

This format allows easy comparison across models for each question.

### Checkpoint Files
`evaluation/results_checkpoint_{model}_{count}.csv` - Saved every 100 queries to prevent data loss.

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors
1. **Reduce batch size**: Lower `BATCH_SIZE` in `evaluate.py`
2. **Increase quantization**: Use 4-bit instead of 8-bit
3. **Lower token limits**: Reduce `max_new_tokens`
4. **Enable CPU offloading**: Already enabled for 4-bit quantization

### Model Download Issues
1. Verify Hugging Face token in `.env`
2. Check internet connection
3. Ensure sufficient disk space
4. Try downloading manually:
```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B --cache-dir /your/path
```

### Version Compatibility Issues
If you encounter errors related to transformers or other packages:

1. **Check installed versions:**
```bash
pip list | grep -E "transformers|torch|accelerate|bitsandbytes"
```

2. **Reinstall with exact versions:**
```bash
pip install --force-reinstall transformers==4.57.3 accelerate==1.12.0 bitsandbytes==0.48.2
```

3. **Common version conflicts:**
- `transformers < 4.30`: Missing 4-bit quantization support
- `bitsandbytes < 0.39`: Incompatible with newer transformers
- `torch < 2.0`: Missing required CUDA features
- `accelerate < 0.20`: Device mapping issues

4. **If problems persist, recreate environment:**
```bash
conda deactivate
conda env remove -n llm-judge
conda create -n llm-judge python=3.10 -y
conda activate llm-judge
pip install -r requirements.txt
```

### Slow Inference
1. Increase `BATCH_SIZE` if you have more RAM
2. Use 4-bit quantization for all models
3. Reduce `max_new_tokens` if responses are too long
4. Check GPU utilization: `watch -n 1 nvidia-smi`

### Resume After Interruption
If evaluation is interrupted:
1. Check latest checkpoint in `evaluation/`
2. Copy checkpoint to `results.csv` if needed
3. Run with `--append` flag to continue from last saved state

### Invalid Scores
If judge returns `None` scores:
- Check judge model output in logs
- Ensure prompt format is correct
- Increase judge's `max_new_tokens` if responses are truncated

## 📝 Adding New Models

1. **Find model on Hugging Face**: e.g., `google/gemma-7b`
2. **Add to MODEL_CONFIGS** in `scripts/evaluate.py`:
```python
MODEL_CONFIGS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "gemma-7b": "google/gemma-7b",  # New model
}
```
3. **Run evaluation**:
```bash
python scripts/evaluate.py --append  # Add to existing results
# OR
python scripts/evaluate.py  # Start fresh
```

## 📂 Project Structure

```
intro-llm-project/
├── data/                          # Dataset files
│   ├── CS-PROB Dataset.xlsx      # Original dataset
│   └── processed_csprob.csv      # Preprocessed data with Q_ID and domains
├── scripts/                       # Python scripts
│   ├── preprocess_dataset.py     # Dataset preprocessing
│   ├── evaluate.py               # Main evaluation pipeline
│   └── download_models.py        # Model download utility
├── evaluation/                    # Results and logs
│   ├── results.csv               # Main results file
│   ├── results_checkpoint_*.csv  # Checkpoint files
│   └── evaluation.log            # Detailed execution logs
├── .env                          # Environment variables (HF_TOKEN)
└── README.md                     # This file
```

## 📞 Contact & Support

For issues or questions about this pipeline, please open an issue in the repository.

---

**Last Updated**: December 3, 2025  
**Pipeline Version**: 1.0  
**Python Version**: 3.10+  
**CUDA Version**: 12.x
