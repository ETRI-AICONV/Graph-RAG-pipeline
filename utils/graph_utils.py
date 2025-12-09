"""
Graph utilities for Hierarchical Graph-RAG pipeline
- Flatten context, get gold indices, build hierarchical graph, KNN edges
- Shared by retriever and evaluation pipeline
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from threading import Lock

# 모델 로딩 동기화를 위한 Lock
_model_lock = Lock()
_model_cache = {}  # (model_name, device) -> encoder

def flatten_context(context):
    """
    Flatten context while preserving document boundary information
    
    Returns:
        sents: List of sentences
        sent_metadata: List of (doc_id, local_sent_id) tuples for each sentence
    """
    sents = []
    sent_metadata = []  # (doc_id, local_sent_id) for each sentence
    for doc_id, doc_sents in enumerate(context['sentences']):
        for local_sent_id, s in enumerate(doc_sents):
            s = s.strip()
            if s:
                sents.append(s)
                sent_metadata.append((doc_id, local_sent_id))
    return sents, sent_metadata


import re
import unicodedata

def normalize_string(s: str) -> str:
    """Normalize string for matching: strip, lower, normalize unicode, remove punctuation"""
    # Strip whitespace
    s = s.strip()
    # Lowercase
    s = s.lower()
    # Normalize unicode (NFKD: decompose then recompose)
    s = unicodedata.normalize('NFKD', s)
    # Remove punctuation (keep alphanumeric and spaces)
    s = re.sub(r'[^\w\s]', '', s)
    # Normalize whitespace (multiple spaces to single)
    s = re.sub(r'\s+', ' ', s)
    return s

def get_gold_indices(context, supporting_facts, sentences_flat, sent_metadata=None):
    """
    Get gold indices using normalized string matching
    
    Args:
        context: Original context with title and sentences
        supporting_facts: Supporting facts with title and sent_id
        sentences_flat: Flattened sentence list
        sent_metadata: Optional list of (doc_id, local_sent_id) tuples
    
    Returns:
        gold_sents: List of indices in sentences_flat
    """
    gold_sents = []
    titles = context['title']
    sentences_list = context['sentences']
    
    # Normalize all flattened sentences once
    normalized_flat = [normalize_string(s) for s in sentences_flat]
    
    for title, sent_id in zip(supporting_facts['title'], supporting_facts['sent_id']):
        if title in titles:
            doc_idx = titles.index(title)
            if sent_id < len(sentences_list[doc_idx]):
                gold = sentences_list[doc_idx][sent_id].strip()
                gold_normalized = normalize_string(gold)
                
                # Try exact match first (faster)
                try:
                    idx = sentences_flat.index(gold)
                    gold_sents.append(idx)
                    continue
                except ValueError:
                    pass
                
                # Fallback to normalized match
                try:
                    idx = normalized_flat.index(gold_normalized)
                    gold_sents.append(idx)
                except ValueError:
                    # If metadata available, use it for direct lookup
                    if sent_metadata is not None:
                        for idx, (doc_id, local_sent_id) in enumerate(sent_metadata):
                            if doc_id == doc_idx and local_sent_id == sent_id:
                                gold_sents.append(idx)
                                break
    
    return gold_sents


def build_edges_knn(sent_emb_cpu, query_emb_cpu, sent_knn=6, query_topk=6, device="cpu"):
    sent = sent_emb_cpu.numpy()
    q = query_emb_cpu.numpy().reshape(-1)
    N = sent.shape[0]
    if N==0:
        return torch.zeros((1,1), dtype=torch.float32, device=device)

    sent_norm = sent / np.linalg.norm(sent, axis=1, keepdims=True).clip(1e-8)
    q_norm = q / np.linalg.norm(q).clip(1e-8)
    sim_matrix = sent_norm @ sent_norm.T
    adj = np.zeros((N+1, N+1), dtype=np.float32)

    for i in range(N):
        row = sim_matrix[i]
        row[i] = -1.0
        k = min(sent_knn, N-1)
        if k>0:
            idxs = np.argpartition(-row, k-1)[:k]
            idxs = idxs[np.argsort(-row[idxs])]
            for j in idxs:
                adj[1+i,1+j] = float(max(row[j],0.0))

    q_sims = sent_norm @ q_norm
    kq = min(query_topk, N)
    if kq>0:
        kth = max(kq-1,0)
        q_idxs = np.argpartition(-q_sims, kth)[:kq]
        q_idxs = q_idxs[np.argsort(-q_sims[q_idxs])]
        for j in q_idxs:
            adj[0,1+j] = float(max(q_sims[j],0.0))

    median_sim = np.median(q_sims) if N>0 else 0.0
    for i in range(N):
        if q_sims[i] >= median_sim:
            adj[1+i,0] = float(q_sims[i])

    # 🔹 torch tensor로 반환
    return torch.tensor(adj, dtype=torch.float32, device=device)


def batch_embed(texts, model_name="all-MiniLM-L6-v2", device="cpu", batch_size=64, show_progress=False):
    # NVML 오류 방지: 단일 GPU만 사용
    if isinstance(device, torch.device) and device.type == "cuda":
        # CUDA device를 문자열로 변환 (예: "cuda:0")
        device_str = str(device)
    else:
        device_str = device
    
    # 모델 캐시 키
    cache_key = (model_name, device_str)
    
    # 모델 로딩 동기화 및 캐싱
    with _model_lock:
        if cache_key not in _model_cache:
            try:
                # 모델 로드 시도
                encoder = SentenceTransformer(model_name, device=device_str)
                _model_cache[cache_key] = encoder
            except RuntimeError as e:
                if "meta tensor" in str(e).lower() or "Cannot copy out of meta tensor" in str(e):
                    # Meta tensor 오류: 모델을 다시 로드
                    print(f"Meta tensor 오류 발생, 모델 재로드 시도: {e}")
                    try:
                        # CPU에서 먼저 로드한 후 GPU로 이동
                        encoder = SentenceTransformer(model_name, device="cpu")
                        if device_str.startswith("cuda"):
                            encoder = encoder.to(device_str)
                        _model_cache[cache_key] = encoder
                    except Exception as e2:
                        print(f"모델 재로드 실패, CPU 사용: {e2}")
                        encoder = SentenceTransformer(model_name, device="cpu")
                        _model_cache[(model_name, "cpu")] = encoder
                elif "NVML" in str(e) or "CUDA" in str(e):
                    # GPU 오류 시 CPU로 fallback
                    print(f"GPU 오류 발생, CPU로 전환: {e}")
                    encoder = SentenceTransformer(model_name, device="cpu")
                    _model_cache[(model_name, "cpu")] = encoder
                else:
                    raise
        
        encoder = _model_cache.get(cache_key)
        if encoder is None:
            # Fallback to CPU
            encoder = _model_cache.get((model_name, "cpu"))
            if encoder is None:
                encoder = SentenceTransformer(model_name, device="cpu")
                _model_cache[(model_name, "cpu")] = encoder
    
    # Embedding 생성 (동기화된 모델 사용)
    try:
        emb = encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=show_progress)
        return torch.tensor(emb)
    except RuntimeError as e:
        if "meta tensor" in str(e).lower() or "Cannot copy out of meta tensor" in str(e):
            # Meta tensor 오류: CPU로 재시도
            print(f"Embedding 중 meta tensor 오류, CPU로 재시도: {e}")
            with _model_lock:
                cpu_key = (model_name, "cpu")
                if cpu_key not in _model_cache:
                    _model_cache[cpu_key] = SentenceTransformer(model_name, device="cpu")
                encoder = _model_cache[cpu_key]
            emb = encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=show_progress)
            return torch.tensor(emb)
        elif "NVML" in str(e) or "CUDA" in str(e):
            # GPU 오류 시 CPU로 fallback
            print(f"GPU 오류 발생, CPU로 전환: {e}")
            with _model_lock:
                cpu_key = (model_name, "cpu")
                if cpu_key not in _model_cache:
                    _model_cache[cpu_key] = SentenceTransformer(model_name, device="cpu")
                encoder = _model_cache[cpu_key]
            emb = encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=show_progress)
            return torch.tensor(emb)
        raise

# ---------------------
# Retrieval + metric helpers
# ---------------------
def retrieve_with_threshold(entry, model, thr=0.5, device='cpu'):
    sent_emb, q_emb, sents = entry['sent_emb'], entry['q_emb'], entry['sents']
    x = torch.cat([q_emb.to(device), sent_emb.to(device)], dim=0)
    adj = build_edges_knn(sent_emb, q_emb).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(adj, x)
        probs = torch.sigmoid(logits)[1:]  # remove query node
    pred_sents = [sents[i] for i in range(len(probs)) if probs[i].item() >= thr]
    return pred_sents

def supporting_fact_f1(pred, gold):
    # CRITICAL: 정규화된 매칭 사용 (공백, 대소문자, 구두점 차이 무시)
    pred_set = set([normalize_string(s) for s in pred])
    gold_set = set([normalize_string(s) for s in gold])
    if not pred_set or not gold_set:
        return 0.0
    inter = len(pred_set & gold_set)
    p, r = inter/len(pred_set), inter/len(gold_set)
    if p+r==0:
        return 0.0
    return 2*p*r/(p+r)

def supporting_fact_em(pred, gold):
    # CRITICAL: 정규화된 매칭 사용 (공백, 대소문자, 구두점 차이 무시)
    pred_set = set([normalize_string(s) for s in pred])
    gold_set = set([normalize_string(s) for s in gold])
    return 1.0 if pred_set == gold_set else 0.0


# ---------------------
# Hierarchical Graph Builder
# ---------------------
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

def clear_gpu_cache():
    """GPU 메모리 정리 헬퍼"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_hierarchical_graph(
    sentences: List[str],
    query: str,
    sent_metadata: Optional[List[Tuple[int, int]]] = None,  # (doc_id, local_sent_id)
    context: Optional[Dict] = None,  # {'title': [...], 'sentences': [[...], ...]}
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    top_k_query_doc: int = None,  # Deprecated: now connects ALL documents
    top_k_query_sent: int = None,  # Deprecated: now connects ALL sentences
    similarity_threshold: float = 0.3,
    neighbor_window: int = 2
) -> Tuple[torch.Tensor, List[str], torch.Tensor, torch.Tensor, List[Dict], int]:
    """
    Build hierarchical graph: [query, doc1, ..., docM, sent1, ..., sentN]
    
    Args:
        sentences: List of all sentences
        query: Query text
        sent_metadata: List of (doc_id, local_sent_id) for each sentence
        context: Original context with 'title' and 'sentences'
        model_name: Embedding model name
        device: Device to use
        top_k_query_doc: Deprecated (not used, all documents are connected)
        top_k_query_sent: Deprecated (not used, all sentences are connected)
        similarity_threshold: Threshold for inter-doc/inter-sent edges
        neighbor_window: Window size for intra-doc neighbors
    
    Returns:
        adj: (1+M+N, 1+M+N) adjacency matrix
        node_texts: [query, doc1, ..., docM, sent1, ..., sentN]
        node_emb: (1+M+N, embedding_dim)
        query_emb: (embedding_dim,) - separate query embedding (for backward compatibility)
        node_metadata: List of dicts with node information
        num_docs: M (number of documents)
    """
    if len(sentences) == 0:
        empty_adj = torch.zeros((0, 0), dtype=torch.float32, device=device)
        empty_emb = torch.zeros((0, 384), dtype=torch.float32, device=device)
        device_str = str(device) if isinstance(device, torch.device) else device
        query_emb = batch_embed([query], model_name=model_name, device=device_str, batch_size=1)[0]
        return empty_adj, [], empty_emb, query_emb, [], 0
    
    device_str = str(device) if isinstance(device, torch.device) else device
    
    # Get document information
    if context is None or 'title' not in context:
        # Fallback: assume single document
        titles = ["Document"]
        num_docs = 1
        if sent_metadata is None:
            sent_metadata = [(0, i) for i in range(len(sentences))]
    else:
        titles = context['title']
        num_docs = len(titles)
        if sent_metadata is None:
            # Reconstruct from context
            sent_metadata = []
            for doc_id, doc_sents in enumerate(context['sentences']):
                for local_sent_id in range(len(doc_sents)):
                    sent_metadata.append((doc_id, local_sent_id))
    
    N = len(sentences)  # Number of sentences
    M = num_docs  # Number of documents
    
    # ==================== Step 1: Embed all nodes ====================
    try:
        if device_str.startswith("cuda"):
            torch.cuda.empty_cache()
        
        # Embed query
        query_emb = batch_embed([query], model_name=model_name, device=device_str, batch_size=1)[0]
        
        # Embed documents (titles)
        doc_texts = titles
        doc_embs = batch_embed(doc_texts, model_name=model_name, device=device_str, batch_size=128)
        
        # Embed sentences
        sent_embs = batch_embed(sentences, model_name=model_name, device=device_str, batch_size=128)
        
    except RuntimeError as e:
        if "CUDA" in str(e) or "NVML" in str(e):
            clear_gpu_cache()
            print(f"GPU 오류로 CPU 전환: {e}")
            query_emb = batch_embed([query], model_name=model_name, device="cpu", batch_size=1)[0]
            doc_embs = batch_embed(doc_texts, model_name=model_name, device="cpu", batch_size=256)
            sent_embs = batch_embed(sentences, model_name=model_name, device="cpu", batch_size=256)
        else:
            raise
    
    # ==================== Step 2: Build node lists ====================
    node_texts = [query] + doc_texts + sentences
    node_emb = torch.cat([query_emb.unsqueeze(0), doc_embs, sent_embs], dim=0)  # (1+M+N, embedding_dim)
    
    # Build node metadata
    node_metadata = []
    node_metadata.append({
        'type': 'query',
        'doc_id': -1,
        'local_sent_id': -1,
        'text': query
    })
    for doc_id, title in enumerate(titles):
        node_metadata.append({
            'type': 'doc',
            'doc_id': doc_id,
            'local_sent_id': -1,
            'text': title
        })
    for sent_idx, (doc_id, local_sent_id) in enumerate(sent_metadata):
        node_metadata.append({
            'type': 'sentence',
            'doc_id': doc_id,
            'local_sent_id': local_sent_id,
            'text': sentences[sent_idx]
        })
    
    # ==================== Step 3: Build adjacency matrix ====================
    adj = torch.zeros((1+M+N, 1+M+N), dtype=torch.float32, device=device)
    
    # Normalize embeddings
    query_emb_norm = query_emb / (torch.norm(query_emb) + 1e-8)
    doc_embs_norm = doc_embs / (torch.norm(doc_embs, dim=1, keepdim=True) + 1e-8)
    sent_embs_norm = sent_embs / (torch.norm(sent_embs, dim=1, keepdim=True) + 1e-8)
    
    # (1) Self-loops for all nodes
    for i in range(1+M+N):
        adj[i, i] = 1.0
    
    # (2) Query ↔ Document edges (ALL documents, no top-k filtering)
    if M > 0:
        query_doc_sim = torch.mm(query_emb_norm.unsqueeze(0), doc_embs_norm.t()).squeeze(0)  # (M,)
        # Connect ALL documents to query (if similarity > 0)
        for idx in range(M):
            doc_node_idx = 1 + idx  # doc nodes start at 1
            sim_val = query_doc_sim[idx].item()
            if sim_val > 0:
                adj[0, doc_node_idx] = sim_val  # query → doc
                adj[doc_node_idx, 0] = sim_val  # doc → query
    
    # (3) Query ↔ Sentence edges (ALL sentences, no top-k filtering)
    if N > 0:
        query_sent_sim = torch.mm(query_emb_norm.unsqueeze(0), sent_embs_norm.t()).squeeze(0)  # (N,)
        # Connect ALL sentences to query (if similarity > 0)
        for idx in range(N):
            sent_node_idx = 1 + M + idx  # sent nodes start at 1+M
            sim_val = query_sent_sim[idx].item()
            if sim_val > 0:
                adj[0, sent_node_idx] = sim_val  # query → sent
                adj[sent_node_idx, 0] = sim_val  # sent → query
    
    # (4) Document ↔ Sentence edges (same document)
    if sent_metadata is not None:
        for sent_idx, (doc_id, local_sent_id) in enumerate(sent_metadata):
            if 0 <= doc_id < M:
                doc_node_idx = 1 + doc_id
                sent_node_idx = 1 + M + sent_idx
                adj[doc_node_idx, sent_node_idx] = 1.0  # doc → sent
                adj[sent_node_idx, doc_node_idx] = 1.0  # sent → doc
    
    # (5) Sentence ↔ Sentence edges
    if N > 0:
        # (A) Intra-doc: consecutive sentences
        if sent_metadata is not None:
            doc_sent_map = defaultdict(list)  # doc_id -> [(global_idx, local_sent_id), ...]
            for global_idx, (doc_id, local_sent_id) in enumerate(sent_metadata):
                doc_sent_map[doc_id].append((global_idx, local_sent_id))
            
            for doc_id, doc_sents in doc_sent_map.items():
                doc_sents_sorted = sorted(doc_sents, key=lambda x: x[1])  # Sort by local_sent_id
                for i in range(len(doc_sents_sorted) - 1):
                    global_i, local_i = doc_sents_sorted[i]
                    global_j, local_j = doc_sents_sorted[i + 1]
                    if abs(local_j - local_i) <= neighbor_window:
                        sent_i_idx = 1 + M + global_i
                        sent_j_idx = 1 + M + global_j
                        adj[sent_i_idx, sent_j_idx] = 1.0
                        adj[sent_j_idx, sent_i_idx] = 1.0
        
        # (B) Inter-doc: Cosine similarity > threshold
        sent_sent_sim = torch.mm(sent_embs_norm, sent_embs_norm.t())  # (N, N)
        for i in range(N):
            for j in range(i+1, N):
                sim_val = sent_sent_sim[i, j].item()
                if sim_val > similarity_threshold:
                    sent_i_idx = 1 + M + i
                    sent_j_idx = 1 + M + j
                    adj[sent_i_idx, sent_j_idx] = sim_val
                    adj[sent_j_idx, sent_i_idx] = sim_val
    
    # (6) Document ↔ Document edges
    if M > 1:
        doc_doc_sim = torch.mm(doc_embs_norm, doc_embs_norm.t())  # (M, M)
        for i in range(M):
            for j in range(i+1, M):
                sim_val = doc_doc_sim[i, j].item()
                if sim_val > 0.2:  # Lower threshold for doc-doc
                    doc_i_idx = 1 + i
                    doc_j_idx = 1 + j
                    adj[doc_i_idx, doc_j_idx] = sim_val
                    adj[doc_j_idx, doc_i_idx] = sim_val
    
    return adj, node_texts, node_emb, query_emb, node_metadata, num_docs