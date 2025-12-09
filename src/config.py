"""
Configuration constants for HotpotQA Graph-RAG Pipeline
"""
import os
import sys

# Add parent directory to path for utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpu_utils import set_device_auto, get_available_gpus

# GPU 설정
AVAILABLE_GPUS = get_available_gpus(min_memory_mb=2000)
DEVICE = set_device_auto(min_memory_mb=2000, use_multi_gpu=False)

# Embedding 모델 설정
MODEL_NAME = "all-MiniLM-L6-v2"  # 빠른 모델 사용
# 옵션: "all-MiniLM-L6-v2" (기본, 384dim, 빠름), "all-mpnet-base-v2" (더 정확하지만 느림)

# Model hyperparameters
HIDDEN_DIM = 256
GNN_LAYERS = 2  # Reduced from 3 to prevent over-smoothing
THRESHOLD = 0.3  # Adaptive threshold will be used if scores are too low

