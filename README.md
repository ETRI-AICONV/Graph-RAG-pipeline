# HotpotQA Graph-RAG Pipeline

Hierarchical Graph-RAG pipeline for HotpotQA question answering.

## Features

- **Hierarchical Graph Construction**: Query + Documents + Sentences
- **Question-aware GAT Retriever**: Graph Attention Network for sentence retrieval
- **T5 Generator**: Sequence-to-sequence answer generation
- **End-to-End Evaluation**: Supporting facts retrieval + Answer generation


## Quick Start

### Environment Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check GPU (optional)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Full Dataset

```bash
# Train with full dataset
python main.py \
    --samples all \
    --retriever_epochs 20 \
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
4. **Generator Training**: Trains generator
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


## Model Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, fast)
- **Retriever**: Hierarchical Query-Aware GAT (2 layers, 256 hidden dim)
- **Generator**: T5-base (220M parameters)

