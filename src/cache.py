"""
HotpotQA Graph-RAG Pipeline - Modular Components
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pickle
import re
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.graph_utils import flatten_context, get_gold_indices, supporting_fact_em, supporting_fact_f1, batch_embed, build_hierarchical_graph
from utils.gpu_utils import set_device_auto, get_available_gpus
from .config import DEVICE, AVAILABLE_GPUS, MODEL_NAME, HIDDEN_DIM, GNN_LAYERS, THRESHOLD

# ==================== Cache Building ====================
def build_hierarchical_graph_cache(split_ds, split_name, cache_path, max_samples=None):
    """Build hierarchical graph cache (query + documents + sentences)"""
    if os.path.exists(cache_path):
        try:
            # Load with device mapping
            with open(cache_path, "rb") as f:
                class CPU_Unpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'torch.storage' and name == '_load_from_bytes':
                            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
                        return super().find_class(module, name)
                cached = CPU_Unpickler(f).load()
            
            cached_size = len(cached)
            expected_size = max_samples if max_samples else len(split_ds)
            
            # 캐시 크기 확인: 정확히 일치해야 함
            # 전체 데이터셋 요청(max_samples=None)일 때는 캐시가 전체 크기와 정확히 일치해야 함
            if cached_size == expected_size:
                print(f"Loading {split_name} cache from {cache_path} ({cached_size} samples)")
                # Move to current device and ensure hierarchical graph fields exist
                for key in cached:
                    if 'adj' in cached[key] and cached[key]['adj'] is not None:
                        cached[key]['adj'] = cached[key]['adj'].to(DEVICE)
                    if 'node_emb' in cached[key] and cached[key]['node_emb'] is not None:
                        cached[key]['node_emb'] = cached[key]['node_emb'].to(DEVICE)
                    # Ensure hierarchical graph fields exist (for backward compatibility)
                    if 'num_docs' not in cached[key]:
                        # Old cache: estimate num_docs from node_emb shape
                        if 'node_emb' in cached[key] and cached[key]['node_emb'] is not None:
                            total_nodes = cached[key]['node_emb'].shape[0]
                            n_sents = len(cached[key].get('sents', []))
                            # Estimate: if node_emb has query, then total = 1 + M + N
                            if total_nodes > n_sents:
                                cached[key]['num_docs'] = max(0, total_nodes - 1 - n_sents)
                            else:
                                cached[key]['num_docs'] = 0
                        else:
                            cached[key]['num_docs'] = 0
                    if 'node_metadata' not in cached[key]:
                        # Create basic metadata for backward compatibility
                        node_texts = cached[key].get('node_texts', [])
                        sent_metadata = cached[key].get('sent_metadata', [])
                        cached[key]['node_metadata'] = []
                        query_offset = 1 if len(node_texts) > len(cached[key].get('sents', [])) else 0
                        for idx, text in enumerate(node_texts):
                            if idx == 0 and query_offset > 0:
                                cached[key]['node_metadata'].append({
                                    'type': 'query', 'doc_id': -1, 'local_sent_id': -1, 'text': text
                                })
                            elif idx < len(node_texts) - len(cached[key].get('sents', [])):
                                cached[key]['node_metadata'].append({
                                    'type': 'doc', 'doc_id': idx - query_offset, 'local_sent_id': -1, 'text': text
                                })
                            else:
                                sent_idx = idx - (len(node_texts) - len(cached[key].get('sents', [])))
                                if sent_idx < len(sent_metadata):
                                    doc_id, local_sent_id = sent_metadata[sent_idx]
                                    cached[key]['node_metadata'].append({
                                        'type': 'sentence', 'doc_id': doc_id, 'local_sent_id': local_sent_id, 'text': text
                                    })
                                else:
                                    cached[key]['node_metadata'].append({
                                        'type': 'sentence', 'doc_id': -1, 'local_sent_id': -1, 'text': text
                                    })
                return cached
            else:
                print(f"Cache size mismatch ({cached_size} vs {expected_size}). Rebuilding...")
                os.remove(cache_path)
        except Exception as e:
            print(f"Error loading cache: {e}. Rebuilding...")
            os.remove(cache_path)
    
    print(f"Building {split_name} cache (hierarchical graph: query + documents + sentences)...")
    
    # Multi-GPU 설정 (오류 시 single-GPU로 fallback)
    available_gpus = AVAILABLE_GPUS.copy() if AVAILABLE_GPUS else []
    use_multi_gpu = False
    num_workers = 1
    
    # Multi-GPU 시도 (샘플 수가 충분할 때만)
    if len(available_gpus) > 1 and (max_samples is None or max_samples > 10):
        try:
            use_multi_gpu = True
            num_workers = min(len(available_gpus), 4)  # 최대 4개 워커
        except Exception as e:
            use_multi_gpu = False
            num_workers = 1
    
    primary_device = DEVICE
    
    cache = {}
    samples = list(range(len(split_ds)))
    if max_samples:
        samples = samples[:max_samples]
    
    # 샘플 처리 함수
    def process_sample(i, gpu_id=None):
        """단일 샘플 처리 함수"""
        try:
            # GPU 설정
            if gpu_id is not None and gpu_id < len(available_gpus):
                sample_device = torch.device(f"cuda:{available_gpus[gpu_id]}")
                torch.cuda.set_device(available_gpus[gpu_id])
            else:
                sample_device = primary_device
                if sample_device.type == "cuda":
                    torch.cuda.set_device(sample_device.index)
            
            s = dict(split_ds[i])
            sents, sent_metadata = flatten_context(s["context"])  # (doc_id, local_sent_id) 정보 포함
            q = s["question"]
            
            if len(sents) == 0:
                return i, None
            
            # Build hierarchical graph: [query, doc1, ..., docM, sent1, ..., sentN]
            try:
                adj, node_texts, node_emb, query_emb, node_metadata, num_docs = build_hierarchical_graph(
                    sentences=sents,
                    query=q,
                    sent_metadata=sent_metadata,
                    context=s["context"],
                    model_name=MODEL_NAME,
                    device=sample_device,
                    top_k_query_doc=10,
                    top_k_query_sent=30,
                    similarity_threshold=0.3,
                    neighbor_window=2
                )
                # node_texts: [query, doc1, ..., docM, sent1, ..., sentN]
                # node_emb: (1+M+N, 384) - hierarchical graph
                # adj: (1+M+N, 1+M+N) - hierarchical adjacency matrix
                # query_emb: (384,) - query embedding (for backward compatibility)
                # node_metadata: List of dicts with node information
                # num_docs: M (number of documents)
            except Exception as e:
                print(f"  Sample {i}: Graph building failed - {e}")
                import traceback
                traceback.print_exc()
                # Create empty graph as fallback
                n_sentences = len(sents)
                if n_sentences > 0:
                    # Fallback: simple graph with query + sentences only
                    adj = torch.eye(1 + n_sentences, dtype=torch.float32, device=sample_device)
                    node_texts = [q] + sents
                    node_emb = torch.randn((1 + n_sentences, 384), dtype=torch.float32, device=sample_device)
                    query_emb = torch.randn((384,), dtype=torch.float32, device=sample_device)
                    node_metadata = [{'type': 'query', 'doc_id': -1, 'local_sent_id': -1, 'text': q}] + \
                                   [{'type': 'sentence', 'doc_id': d, 'local_sent_id': l, 'text': s} 
                                    for (d, l), s in zip(sent_metadata, sents)]
                    num_docs = 0
                else:
                    adj = torch.zeros((0, 0), dtype=torch.float32, device=sample_device)
                    node_texts = []
                    node_emb = torch.zeros((0, 384), dtype=torch.float32, device=sample_device)
                    query_emb = torch.randn((384,), dtype=torch.float32, device=sample_device)
                    node_metadata = []
                    num_docs = 0
            
            # Gold indices with normalized matching and metadata
            gold_idx = get_gold_indices(s["context"], s["supporting_facts"], sents, sent_metadata)
            
            # Move tensors to CPU before caching (to avoid device mismatch)
            result = {
                "sents": sents,  # Original sentences (for backward compatibility)
                "adj": adj.cpu() if isinstance(adj, torch.Tensor) else adj,
                "node_texts": node_texts,  # [query, doc1, ..., docM, sent1, ..., sentN]
                "node_emb": node_emb.cpu() if isinstance(node_emb, torch.Tensor) else node_emb,
                "query_emb": query_emb.cpu() if isinstance(query_emb, torch.Tensor) else query_emb,
                "gold": gold_idx,  # Indices in sents (original sentence indices)
                "question": q,
                "answer": s["answer"],
                "sent_metadata": sent_metadata,  # (doc_id, local_sent_id) for each sentence
                "node_metadata": node_metadata,  # Full node metadata for hierarchical graph
                "num_docs": num_docs  # M (number of documents)
            }
            
            return i, result
                
        except Exception as e:
            print(f"  Sample {i}: Processing failed - {e}")
            import traceback
            traceback.print_exc()
            return i, None
    
    # 병렬 처리 실행 (Multi-GPU) 또는 순차 처리 (Single-GPU)
    if use_multi_gpu and num_workers > 1:
        print(f"Processing {len(samples)} samples in parallel using {num_workers} workers...")
        cache_lock = Lock()
        completed = 0
        failed_count = 0
        
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 모든 작업 제출 (GPU 분산)
                future_to_sample = {}
                for idx, sample_idx in enumerate(samples):
                    gpu_id = idx % num_workers  # Round-robin GPU 할당
                    future = executor.submit(process_sample, sample_idx, gpu_id)
                    future_to_sample[future] = sample_idx
                
                # 완료된 작업 처리
                for future in tqdm(as_completed(future_to_sample), total=len(samples), desc=f"Processing {split_name}"):
                    sample_idx, result = future.result()
                    if result is not None:
                        with cache_lock:
                            cache[sample_idx] = result
                            completed += 1
                            # 주기적으로 메모리 정리
                            if completed % 5 == 0:
                                for gpu_id in range(num_workers):
                                    try:
                                        if gpu_id < len(available_gpus):
                                            torch.cuda.set_device(available_gpus[gpu_id])
                                            torch.cuda.empty_cache()
                                    except:
                                        pass
                    else:
                        failed_count += 1
                        if failed_count > len(samples) * 0.5:  # 50% 이상 실패 시 중단
                            raise Exception("Multi-GPU processing failed")
        except Exception as e:
            use_multi_gpu = False
            # 실패한 샘플들을 single-GPU로 재처리
            remaining_samples = [i for i in samples if i not in cache]
            if remaining_samples:
                for i in tqdm(remaining_samples, desc=f"Retrying {split_name}"):
                    sample_idx, result = process_sample(i, None)
                    if result is not None:
                        cache[sample_idx] = result
                    if len(cache) % 5 == 0 and primary_device.type == "cuda":
                        torch.cuda.empty_cache()
    
    # Single-GPU 모드: 순차 처리
    if not use_multi_gpu:
        for i in tqdm(samples, desc=f"Processing {split_name}"):
            sample_idx, result = process_sample(i, None)
            if result is not None:
                cache[sample_idx] = result
            # 주기적으로 메모리 정리
            if len(cache) % 5 == 0 and primary_device.type == "cuda":
                torch.cuda.empty_cache()
    
    # Save cache
    print(f"Saving cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    
    # Reload and move to DEVICE
    with open(cache_path, "rb") as f:
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
                return super().find_class(module, name)
        cache = CPU_Unpickler(f).load()
    
    # Move to current device
    for key in cache:
        if 'adj' in cache[key] and cache[key]['adj'] is not None:
            cache[key]['adj'] = cache[key]['adj'].to(DEVICE)
        if 'node_emb' in cache[key] and cache[key]['node_emb'] is not None:
            cache[key]['node_emb'] = cache[key]['node_emb'].to(DEVICE)
        if 'query_emb' in cache[key] and cache[key]['query_emb'] is not None:
            cache[key]['query_emb'] = cache[key]['query_emb'].to(DEVICE)
    
    return cache



def build_labels(n_sentences, gold_idx, node_texts, sents, num_docs=0, debug=False):
    """
    Map gold sentences to node indices (hierarchical graph: [query, doc1, ..., docM, sent1, ..., sentN])
    
    Args:
        n_sentences: Number of sentences (N)
        gold_idx: Gold indices in sents list (original sentence indices)
        node_texts: [query, doc1, ..., docM, sent1, ..., sentN] (hierarchical graph)
        sents: Original sentence list [sent1, sent2, ..., sentN]
        num_docs: Number of documents (M)
        debug: If True, print debugging information
    
    Returns:
        y: (n_sentences,) labels - labels for sentence nodes only
    """
    y = torch.zeros(n_sentences, device=DEVICE)
    
    # Hierarchical graph structure:
    # node_texts[0] = query
    # node_texts[1:1+M] = documents
    # node_texts[1+M:1+M+N] = sentences
    
    # gold_idx refers to indices in sents list (original sentence indices)
    # We need to map these to the sentence node indices in node_texts
    # Sentence nodes in node_texts start at index 1+M
    sent_node_start = 1 + num_docs  # Sentence nodes start here
    
    if debug:
        print(f"\n{'='*80}")
        print(f"[build_labels Debug]")
        print(f"{'='*80}")
        print(f"  n_sentences: {n_sentences}, num_docs: {num_docs}")
        print(f"  len(sents): {len(sents)}, len(node_texts): {len(node_texts)}")
        print(f"  sent_node_start: {sent_node_start} (sentence nodes start at this index in node_texts)")
        print(f"  gold_idx: {gold_idx} (indices in sents list)")
        print()
    
    # Verify that node_texts[sent_node_start:] matches sents
    if len(node_texts) >= sent_node_start + n_sentences and len(sents) == n_sentences:
        node_sents = node_texts[sent_node_start:sent_node_start + n_sentences]
        mismatches = []
        for i, (ns, s) in enumerate(zip(node_sents, sents)):
            if ns != s:
                mismatches.append(i)
        
        if debug:
            if len(mismatches) > 0:
                print(f"  WARNING:  WARNING: {len(mismatches)} mismatches between node_texts sentences and sents!")
                print(f"  Mismatch indices: {mismatches[:10]} (showing first 10)")
                for i in mismatches[:3]:  # Show first 3 mismatches
                    print(f"    Mismatch at index {i}:")
                    print(f"      node_texts[{sent_node_start + i}]: {node_sents[i][:80]}...")
                    print(f"      sents[{i}]: {sents[i][:80]}...")
            else:
                print(f"  OK: node_texts sentences and sents match perfectly!")
            
            # Show first few sentences for verification
            print(f"\n  First 3 sentences comparison:")
            for i in range(min(3, len(sents))):
                node_idx = sent_node_start + i
                match = "OK:" if node_idx < len(node_texts) and node_texts[node_idx] == sents[i] else "ERROR:"
                print(f"    [{i}] {match} sents[{i}] == node_texts[{node_idx}]")
                if node_idx < len(node_texts):
                    print(f"        sents: {sents[i][:60]}...")
                    print(f"        node_texts: {node_texts[node_idx][:60]}...")
            print()
    
    # Map gold indices: gold_idx[i] is an index in sents
    # sents[gold_idx[i]] should correspond to node_texts[sent_node_start + gold_idx[i]]
    mapped_count = 0
    not_found_count = 0
    
    if debug:
        print(f"  Mapping {len(gold_idx)} gold indices to labels:")
        print()
    
    for gi in gold_idx:
        if gi < len(sents):
            gold_sent = sents[gi]
            # Direct mapping: sentence at index gi in sents should be at node_texts[sent_node_start + gi]
            if gi < n_sentences:
                node_idx = sent_node_start + gi
                if node_idx < len(node_texts):
                    # Verify by text matching
                    if node_texts[node_idx] == gold_sent:
                        y[gi] = 1.0
                        mapped_count += 1
                        if debug:
                            print(f"  OK: Gold index {gi} → Label {gi} (node {node_idx})")
                            print(f"      Text: {gold_sent[:70]}...")
                    else:
                        # Text doesn't match, search for the correct node
                        found = False
                        for sent_node_idx in range(sent_node_start, len(node_texts)):
                            if node_texts[sent_node_idx] == gold_sent:
                                # Map back to label index
                                label_idx = sent_node_idx - sent_node_start
                                if label_idx < n_sentences:
                                    y[label_idx] = 1.0
                                    found = True
                                    mapped_count += 1
                                    if debug:
                                        print(f"  WARNING:  Gold index {gi} → Label {label_idx} (node {sent_node_idx}, reordered)")
                                        print(f"      Text: {gold_sent[:70]}...")
                                break
                        if not found:
                            not_found_count += 1
                            if debug:
                                print(f"  ERROR: Gold index {gi} NOT FOUND in node_texts!")
                                print(f"      Looking for: {gold_sent[:70]}...")
                                # Show what's at the expected position
                                if node_idx < len(node_texts):
                                    print(f"      Found at node {node_idx}: {node_texts[node_idx][:70]}...")
        else:
            if debug:
                print(f"  WARNING:  Gold index {gi} is out of range (len(sents)={len(sents)})")
    
    if debug:
        print()
        print(f"  Summary:")
        print(f"    Total gold indices: {len(gold_idx)}")
        print(f"    Successfully mapped: {mapped_count}")
        print(f"    Not found: {not_found_count}")
        print(f"    Positive labels created: {y.sum().item()}")
        print(f"    Expected positive labels: {len(gold_idx)}")
        if y.sum().item() != len(gold_idx):
            print(f"    WARNING:  WARNING: Label count mismatch!")
        print(f"{'='*80}\n")
    
    return y
