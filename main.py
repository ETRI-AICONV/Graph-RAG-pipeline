"""
HotpotQA Graph-RAG 통합 파이프라인
- 그래프 구축: Hierarchical graph (query + documents + sentences)
- Retriever: Question-aware GAT
- Generator: T5-base (internal model name)
- 평가: Sup (retrieval), Answer (gold→gen), Joint (E2E)

Usage:
    python main.py --samples 100 --retriever_epochs 20 --generator_epochs 3
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import glob
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from src.config import DEVICE, AVAILABLE_GPUS, MODEL_NAME, HIDDEN_DIM, GNN_LAYERS, THRESHOLD
from src.models import HierarchicalQueryAwareGATRetriever
from src.cache import build_hierarchical_graph_cache, build_labels
from src.losses import WeightedRankingLoss
from src.retriever import train_hierarchical_gat_retriever, retrieve_supporting_facts
from src.generator import train_t5_generator, prepare_t5_training_data
from src.evaluation import evaluate_hierarchical_graph_rag_pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==================== Main Pipeline ====================
def main():
    # Get script directory for absolute paths (public version)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="HotpotQA Graph-RAG Pipeline")
    parser.add_argument("--samples", type=str, default=None, 
                        help="Number of samples to process (default: all, or use 'all' for full dataset)")
    parser.add_argument("--retriever_epochs", type=int, default=20, 
                        help="Number of training epochs for retriever")
    parser.add_argument("--generator_epochs", type=int, default=3, 
                        help="Number of training epochs for generator")
    parser.add_argument("--skip_cache", action="store_true", 
                        help="Skip cache and rebuild (force new cache creation)")
    parser.add_argument("--skip_retriever", action="store_true",
                        help="Skip retriever training (use existing model)")
    parser.add_argument("--skip_generator", action="store_true",
                        help="Skip generator training (use existing model)")
    # Public version: Add path arguments
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory (default: <script_dir>/cache)")
    parser.add_argument("--models_dir", type=str, default=None,
                        help="Models directory (default: <script_dir>/models)")
    
    args = parser.parse_args()
    
    # Set up paths (absolute paths based on script directory)
    if args.cache_dir:
        CACHE_DIR = os.path.abspath(args.cache_dir)
    else:
        CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
    
    if args.models_dir:
        MODELS_DIR = os.path.abspath(args.models_dir)
    else:
        MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
    
    print("="*80)
    print("HotpotQA Graph-RAG Pipeline (Sentence-level Graph)")
    print("="*80)
    
    # Load dataset
    print("\n[1/6] Loading HotpotQA distractor dataset...")
    dataset = load_dataset("hotpot_qa", "distractor")
    
    # Split into train:val:test = 8:1:1
    # First split: 80% train, 20% temp (val+test)
    train_temp = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_ds = train_temp["train"]  # 80%
    temp_ds = train_temp["test"]    # 20%
    
    # Second split: temp를 val과 test로 1:1 분할 (각각 10%)
    val_test = temp_ds.train_test_split(test_size=0.5, seed=42)
    val_ds = val_test["train"]      # 10%
    test_ds = val_test["test"]       # 10%
    
    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print(f"Split ratio: {len(train_ds)/len(dataset['train'])*100:.1f}% : {len(val_ds)/len(dataset['train'])*100:.1f}% : {len(test_ds)/len(dataset['train'])*100:.1f}%")
    
    # Parse samples argument
    num_samples = None
    if args.samples:
        if args.samples.lower() == "all":
            num_samples = None  # None means process all samples
            print(f"Processing all samples from each split (Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)})")
        else:
            try:
                num_samples = int(args.samples)
                print(f"Processing {num_samples} samples from each split")
            except ValueError:
                print(f"Error: --samples must be a number or 'all', got '{args.samples}'")
                sys.exit(1)
    else:
        print(f"Processing all samples from each split (Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)})")
    
    # Build cache (sentence-level graph only)
    print(f"\n[2/6] Building sentence-level graph cache...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    if args.skip_cache:
        # Find latest cache files (with timestamp)
        import glob
        cache_train_files = sorted(glob.glob(os.path.join(CACHE_DIR, "train_cache_*.pkl")))
        cache_val_files = sorted(glob.glob(os.path.join(CACHE_DIR, "val_cache_*.pkl")))
        cache_test_files = sorted(glob.glob(os.path.join(CACHE_DIR, "test_cache_*.pkl")))
        
        cache_train = cache_train_files[-1] if cache_train_files else os.path.join(CACHE_DIR, "train_cache.pkl")
        cache_val = cache_val_files[-1] if cache_val_files else os.path.join(CACHE_DIR, "val_cache.pkl")
        cache_test = cache_test_files[-1] if cache_test_files else os.path.join(CACHE_DIR, "test_cache.pkl")
        t5_out_dir = os.path.join(MODELS_DIR, "t5_generator")
        retriever_model_path = os.path.join(MODELS_DIR, "hierarchical_graph_rag_retriever.pt")
        
        # Remove cache files (only if they exist and --skip_cache is used)
        # Note: With timestamp-based naming, old cache files won't be overwritten
        if os.path.exists(cache_train):
            os.remove(cache_train)
            print(f"Removed cache: {cache_train}")
        if os.path.exists(cache_val):
            os.remove(cache_val)
            print(f"Removed cache: {cache_val}")
        if os.path.exists(cache_test):
            os.remove(cache_test)
            print(f"Removed cache: {cache_test}")
        
        # Remove Generator model directory
        if os.path.exists(t5_out_dir):
            import shutil
            shutil.rmtree(t5_out_dir)
            print(f"Removed Generator model directory: {t5_out_dir}")
        
        # Remove retriever model
        if os.path.exists(retriever_model_path):
            os.remove(retriever_model_path)
            print(f"Removed retriever model: {retriever_model_path}")
    
    # Build hierarchical graph cache: use existing if available, otherwise create new with timestamp
    from datetime import datetime
    import glob
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Find existing cache files (with timestamp)
    existing_train = sorted(glob.glob(os.path.join(CACHE_DIR, "train_cache_*.pkl")))
    existing_val = sorted(glob.glob(os.path.join(CACHE_DIR, "val_cache_*.pkl")))
    existing_test = sorted(glob.glob(os.path.join(CACHE_DIR, "test_cache_*.pkl")))
    
    if args.skip_cache or not existing_train or not existing_val or not existing_test:
        # Create new cache with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_cache_path = os.path.join(CACHE_DIR, f"train_cache_{timestamp}.pkl")
        val_cache_path = os.path.join(CACHE_DIR, f"val_cache_{timestamp}.pkl")
        test_cache_path = os.path.join(CACHE_DIR, f"test_cache_{timestamp}.pkl")
        
        print(f"\n[Cache Files] Creating new cache files with timestamp")
        
        train_cache = build_hierarchical_graph_cache(train_ds, "train", train_cache_path, num_samples)
        val_cache = build_hierarchical_graph_cache(val_ds, "val", val_cache_path, num_samples)
        test_cache = build_hierarchical_graph_cache(test_ds, "test", test_cache_path, num_samples)
    else:
        # Use latest existing cache files
        train_cache_path = existing_train[-1]
        val_cache_path = existing_val[-1]
        test_cache_path = existing_test[-1]
        
        print(f"\n[Cache Files] Using existing cache files")
        
        train_cache = build_hierarchical_graph_cache(train_ds, "train", train_cache_path, num_samples)
        val_cache = build_hierarchical_graph_cache(val_ds, "val", val_cache_path, num_samples)
        test_cache = build_hierarchical_graph_cache(test_ds, "test", test_cache_path, num_samples)
    
    print(f"Cache sizes: Train={len(train_cache)}, Val={len(val_cache)}, Test={len(test_cache)}")
    
    # Train retriever
    # Skip training if retriever_epochs is 0 or skip_retriever flag is set
    if args.retriever_epochs > 0 and not args.skip_retriever:
        print(f"\n[3/6] Training Hierarchical GAT Retriever...")
        retriever_model = train_hierarchical_gat_retriever(train_cache, val_cache=val_cache, num_epochs=args.retriever_epochs)
        os.makedirs(MODELS_DIR, exist_ok=True)
        retriever_model_path = os.path.join(MODELS_DIR, "hierarchical_graph_rag_retriever.pt")
        torch.save(retriever_model.state_dict(), retriever_model_path)
        print(f"Retriever saved to {retriever_model_path}")
    else:
        print(f"\n[3/6] Loading existing retriever model...")
        # Find embedding dimension from cache
        sample_emb_dim = None
        for entry in train_cache.values():
            if "node_emb" in entry and entry["node_emb"] is not None and len(entry["node_emb"]) > 0:
                sample_emb_dim = entry["node_emb"].shape[1]
                break
        if sample_emb_dim is None:
            raise ValueError("Cannot determine embedding dimension from cache!")
        retriever_model = HierarchicalQueryAwareGATRetriever(embedding_dim=sample_emb_dim).to(DEVICE)
        model_path = os.path.join(MODELS_DIR, "hierarchical_graph_rag_retriever.pt")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please train the retriever first or check the model path.")
            sys.exit(1)
        retriever_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        retriever_model.eval()
        print(f"Retriever model loaded from {model_path}")
    
    # Free GPU memory: Move retriever to CPU, move cache to CPU, and clear cache
    retriever_model = retriever_model.cpu()
    for key in train_cache:
        if 'adj' in train_cache[key] and train_cache[key]['adj'] is not None:
            train_cache[key]['adj'] = train_cache[key]['adj'].cpu()
        if 'node_emb' in train_cache[key] and train_cache[key]['node_emb'] is not None:
            train_cache[key]['node_emb'] = train_cache[key]['node_emb'].cpu()
        if 'query_emb' in train_cache[key] and train_cache[key]['query_emb'] is not None:
            train_cache[key]['query_emb'] = train_cache[key]['query_emb'].cpu()
    
    for key in val_cache:
        if 'adj' in val_cache[key] and val_cache[key]['adj'] is not None:
            val_cache[key]['adj'] = val_cache[key]['adj'].cpu()
        if 'node_emb' in val_cache[key] and val_cache[key]['node_emb'] is not None:
            val_cache[key]['node_emb'] = val_cache[key]['node_emb'].cpu()
        if 'query_emb' in val_cache[key] and val_cache[key]['query_emb'] is not None:
            val_cache[key]['query_emb'] = val_cache[key]['query_emb'].cpu()
    
    # Aggressive GPU memory cleanup before Generator training
    if torch.cuda.is_available():
        import gc
        # Move retriever to CPU (don't delete, we need it for evaluation later)
        retriever_model = retriever_model.cpu()
        # Force garbage collection
        gc.collect()
        # Clear CUDA cache multiple times
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Prepare Generator training data
    print(f"\n[4/6] Preparing Generator training data (using GOLD supporting facts)...")
    train_t5_data = prepare_t5_training_data(train_cache, mode="gold")
    val_t5_data = prepare_t5_training_data(val_cache, mode="gold")
    print(f"Generator data: Train={len(train_t5_data)}, Val={len(val_t5_data)}")
    
    # Train generator
    print(f"\n[5/6] Training Generator...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Additional memory cleanup right before Generator training
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    t5_output_dir = os.path.join(MODELS_DIR, "t5_generator")
    generator_model, generator_tokenizer = train_t5_generator(
        train_t5_data, val_t5_data, 
        model_name="t5-base",
        output_dir=t5_output_dir,
        num_epochs=args.generator_epochs,
        skip_training=(args.generator_epochs == 0),  # Skip training only if epochs is 0
        batch_size=32  # Batch size for Generator
    )
    print(f"Generator saved to {t5_output_dir}/")
    
    # Ensure generator_model and generator_tokenizer are set (from training or fallback)
    if generator_model is None or generator_tokenizer is None:
        print(f"WARNING:  Generator model not returned from training, will try to load from checkpoint")
    
    # Move retriever back to GPU for evaluation (after generator training)
    retriever_model = retriever_model.to(DEVICE)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Checkpoint가 있으면 명시적으로 로드
    import glob
    generator_path = os.path.join(MODELS_DIR, "t5_generator")
    
    # Ensure generator_model and generator_tokenizer are defined
    generator_model = None
    generator_tokenizer = None
    
    checkpoints = glob.glob(f"{generator_path}/checkpoint-*")
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"\n[Loading Generator from Checkpoint]")
        try:
            # Load Generator model from checkpoint
            # Check if checkpoint has config.json and it's a valid T5 config
            checkpoint_config = os.path.join(latest_checkpoint, "config.json")
            if os.path.exists(checkpoint_config):
                # Verify config is T5 config before loading
                import json
                with open(checkpoint_config, 'r') as f:
                    config_data = json.load(f)
                    model_type = config_data.get('model_type', '')
                    if model_type not in ['t5', 'mt5', 'umt5']:
                        raise ValueError(f"Invalid model type in checkpoint: {model_type}. Expected Generator model.")
                
                generator_model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint).to(DEVICE)
                generator_tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
                print(f"  OK: Loaded generator from checkpoint")
            else:
                # Try loading from parent directory
                parent_dir = os.path.dirname(latest_checkpoint)
                parent_config = os.path.join(parent_dir, "config.json")
                if os.path.exists(parent_config):
                    import json
                    with open(parent_config, 'r') as f:
                        config_data = json.load(f)
                        model_type = config_data.get('model_type', '')
                        if model_type not in ['t5', 'mt5', 'umt5']:
                            raise ValueError(f"Invalid model type: {model_type}. Expected Generator model.")
                    
                    generator_model = AutoModelForSeq2SeqLM.from_pretrained(parent_dir).to(DEVICE)
                    generator_tokenizer = AutoTokenizer.from_pretrained(parent_dir)
                    print(f"  OK: Loaded generator from parent directory")
                else:
                    raise FileNotFoundError(f"config.json not found in {latest_checkpoint} or {parent_dir}")
        except Exception as e:
            print(f"  WARNING:  Failed to load from checkpoint: {e}")
            print(f"  Trying to load from output directory: {generator_path}")
            try:
                # Verify config before loading
                output_config = os.path.join(generator_path, "config.json")
                if os.path.exists(output_config):
                    import json
                    with open(output_config, 'r') as f:
                        config_data = json.load(f)
                        model_type = config_data.get('model_type', '')
                        if model_type not in ['t5', 'mt5', 'umt5']:
                            raise ValueError(f"Invalid model type: {model_type}. Expected Generator model.")
                
                generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_path).to(DEVICE)
                generator_tokenizer = AutoTokenizer.from_pretrained(generator_path)
                print(f"  OK: Loaded generator from output directory")
            except Exception as e2:
                print(f"  ERROR: Failed to load from output directory: {e2}")
                print(f"  WARNING:  Generator model may not be available for evaluation!")
    
    # Generator 모델 체크 - 모델이 로드되었는지 확인
    if generator_model is None or generator_tokenizer is None:
        print(f"\n{'='*80}")
        print(f"WARNING:  WARNING: Generator model not loaded!")
        print(f"{'='*80}")
        print(f"Generator model and tokenizer must be loaded before evaluation.")
        print(f"Please check:")
        print(f"  1. Model training completed successfully")
        print(f"  2. Model saved to {generator_path}")
        print(f"  3. Checkpoint files exist")
        print(f"{'='*80}\n")
        # Try to load from default path as last resort
        try:
            print(f"Attempting to load from default path: {generator_path}")
            generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_path).to(DEVICE)
            generator_tokenizer = AutoTokenizer.from_pretrained(generator_path)
            print(f"OK: Loaded generator from default path")
        except Exception as e:
            print(f"ERROR: Failed to load generator: {e}")
            print(f"WARNING:  Evaluation will be skipped!")
            return
    
    # Evaluation
    print(f"\n[6/6] Evaluating Full Pipeline...")
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Evaluation on TEST set only (validation is for loss calculation only)
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    # Move retriever back to GPU for evaluation (if needed)
    # Check if retriever is on CPU (by checking first parameter's device)
    retriever_device = next(retriever_model.parameters()).device
    if retriever_device.type != 'cuda':
        retriever_model = retriever_model.to(DEVICE)
    
    # 1. Retriever Performance: Supporting Facts Retrieval
    print("\n[1] Retriever Performance (Supporting Facts Retrieval):")
    retriever_results = evaluate_hierarchical_graph_rag_pipeline(test_cache, retriever_model, generator_model, generator_tokenizer, mode="sup")
    print(f"    Sup EM: {retriever_results['sup_em']:.4f}, Sup F1: {retriever_results['sup_f1']:.4f}")
    print(f"    Hit@1: {retriever_results['hit@1']:.4f}, Hit@3: {retriever_results['hit@3']:.4f}, Hit@5: {retriever_results['hit@5']:.4f}, Hit@10: {retriever_results['hit@10']:.4f}")
    
    # 2. Generator Performance: Answer Generation with GOLD facts
    print("\n[2] Generator Performance (Answer Generation with Gold Facts):")
    generator_results = evaluate_hierarchical_graph_rag_pipeline(test_cache, retriever_model, generator_model, generator_tokenizer, mode="answer")
    print(f"    Ans EM: {generator_results['ans_em']:.4f}, Ans F1: {generator_results['ans_f1']:.4f}")
    
    # 3. Joint Performance: End-to-End with RETRIEVED facts
    print("\n[3] Joint Performance (End-to-End with Retrieved Facts):")
    joint_results = evaluate_hierarchical_graph_rag_pipeline(test_cache, retriever_model, generator_model, generator_tokenizer, mode="joint")
    print(f"    Joint EM: {joint_results['ans_em']:.4f}, Joint F1: {joint_results['ans_f1']:.4f}")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()

