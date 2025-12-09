# HotpotQA Graph-RAG Pipeline

Hierarchical Graph-RAG pipeline for HotpotQA question answering.

## Features

- **Hierarchical Graph Construction**: Query + Documents + Sentences
- **Question-aware GAT Retriever**: Graph Attention Network for sentence retrieval
- **T5 Generator**: Sequence-to-sequence answer generation
- **End-to-End Evaluation**: Supporting facts retrieval + Answer generation

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd hotpot_git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify GPU availability (optional but recommended)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Environment Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check GPU (optional)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Quick Test (100 samples)

```bash
# Quick test with minimal settings
python main.py \
    --samples 100 \
    --retriever_epochs 5 \
    --generator_epochs 2
```

**Estimated time**: ~30-60 minutes on GPU

### Full Dataset

```bash
# Train with full dataset
python main.py \
    --samples all \
    --retriever_epochs 20 \
    --generator_epochs 3
```

**Estimated time**: Several hours on GPU

### Custom Paths

```bash
# Use custom cache and models directories
python main.py \
    --samples 100 \
    --cache_dir /path/to/cache \
    --models_dir /path/to/models \
    --retriever_epochs 10 \
    --generator_epochs 3
```

## Command Line Arguments

| Option | Description | Default |
|--------|-------------|---------|
| `--samples` | Number of samples (`all` or number) | `all` |
| `--retriever_epochs` | Retriever training epochs | `20` |
| `--generator_epochs` | Generator training epochs | `3` |
| `--cache_dir` | Cache directory path | `<script_dir>/cache` |
| `--models_dir` | Models directory path | `<script_dir>/models` |
| `--skip_cache` | Skip cache and rebuild | `False` |
| `--skip_retriever` | Skip retriever training | `False` |
| `--skip_generator` | Skip generator training | `False` |

## Pipeline Steps

1. **Dataset Loading**: Loads HotpotQA distractor dataset (auto-downloads from HuggingFace)
2. **Graph Cache Building**: Builds hierarchical graph cache (query + documents + sentences)
3. **Retriever Training**: Trains Question-aware GAT retriever
4. **Generator Training**: Trains T5 generator
5. **Evaluation**: Evaluates on test set (retrieval + answer generation)

## Directory Structure

```
hotpot_git/
├── main.py                             # Main pipeline script
├── graph_utils.py                      # Graph utilities
├── gpu_utils.py                        # GPU auto-selection utilities
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── cache/                              # Cache files (auto-created, git-ignored)
│   ├── train_cache_*.pkl
│   ├── val_cache_*.pkl
│   └── test_cache_*.pkl
└── models/                             # Trained models (auto-created, git-ignored)
    ├── hierarchical_graph_rag_retriever.pt
    └── t5_generator/
        └── ...
```

## Requirements

### Hardware

- **GPU**: Recommended (minimum 2GB VRAM)
  - Multi-GPU supported automatically
  - CPU mode available but slow
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Disk**: ~10GB for models and cache

### Software

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- See `requirements.txt` for Python packages

## First Run

On first run, the pipeline will:

1. **Download HotpotQA dataset** (~100MB) from HuggingFace
2. **Download embedding model** (`all-MiniLM-L6-v2`, ~80MB)
3. **Download T5-base model** (~850MB)
4. **Build graph cache** (takes time, depends on sample size)
5. **Train models** (takes time, depends on epochs and GPU)

**Estimated time for 100 samples:**
- Cache building: ~5-10 minutes
- Retriever training: ~10-20 minutes
- Generator training: ~15-30 minutes
- **Total: ~30-60 minutes** (on GPU)

## Output

The pipeline outputs:

1. **Cache files**: Pre-computed graph structures (`.pkl` files)
2. **Retriever model**: `models/hierarchical_graph_rag_retriever.pt`
3. **Generator model**: `models/t5_generator/`
4. **Evaluation results**: Printed to console
   - Supporting Facts EM/F1
   - Hit@K metrics
   - Answer EM/F1 (with gold facts)
   - Joint EM/F1 (with retrieved facts)

## Troubleshooting

### GPU Issues

If GPU is not detected:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU devices
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

The pipeline will automatically fall back to CPU if GPU is not available (but will be slow).

### Memory Issues

If you encounter out-of-memory errors:

```bash
# Reduce sample count
python main.py --samples 50

# Or reduce epochs
python main.py --samples 100 --retriever_epochs 5 --generator_epochs 1
```

### Cache Issues

If cache is corrupted:

```bash
# Rebuild cache
python main.py --skip_cache --samples 100
```

## Model Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, fast)
- **Retriever**: Hierarchical Query-Aware GAT (2 layers, 256 hidden dim)
- **Generator**: T5-base (220M parameters)

## Citation

If you use this code, please cite:

```bibtex
@misc{hotpot-graph-rag,
  title={HotpotQA Graph-RAG Pipeline},
  author={Your Name},
  year={2024}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]
