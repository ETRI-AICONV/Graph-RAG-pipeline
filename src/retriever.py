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
from .models import HierarchicalQueryAwareGATRetriever
from .losses import WeightedRankingLoss, ContrastiveLoss
from .cache import build_labels

# Loss classes are now imported from .losses, keeping old definitions for reference only
# (These will be removed in future cleanup)
class ContrastiveLoss(nn.Module):
    """InfoNCE-based Contrastive Loss for Retrieval
    
    Positive와 Negative를 명확히 구분하도록 학습
    InfoNCE: log(exp(pos_score) / (exp(pos_score) + sum(exp(neg_scores))))
    """
    
    def __init__(self, temperature=0.07, pos_weight=1.0):
        """
        Args:
            temperature: Temperature for softmax (낮을수록 더 sharp한 분포)
            pos_weight: Positive sample weight
        """
        super().__init__()
        self.temperature = temperature
        self.pos_weight = pos_weight
    
    def forward(self, scores, labels):
        """
        scores: (N,) 각 문장의 supporting fact 점수
        labels: (N,) 1.0 for positive, 0.0 for negative
        
        Returns: contrastive loss
        """
        if scores.shape[0] != labels.shape[0]:
            raise ValueError(f"Size mismatch: scores.shape={scores.shape}, labels.shape={labels.shape}")
        
        pos_mask = labels > 0.5
        neg_mask = labels < 0.5
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        pos_scores = scores[pos_mask]  # (num_pos,)
        neg_scores = scores[neg_mask]  # (num_neg,)
        
        # InfoNCE Loss: 각 positive에 대해
        # loss = -log(exp(pos_score / temp) / (exp(pos_score / temp) + sum(exp(neg_score / temp))))
        losses = []
        for pos_score in pos_scores:
            # Positive score (scaled by temperature)
            pos_exp = torch.exp(pos_score / self.temperature)  # scalar
            
            # Negative scores (scaled by temperature)
            neg_exps = torch.exp(neg_scores / self.temperature)  # (num_neg,)
            neg_sum = neg_exps.sum()  # scalar
            
            # InfoNCE loss for this positive
            denominator = pos_exp + neg_sum
            loss = -torch.log(pos_exp / denominator + 1e-8)  # Negative log likelihood
            losses.append(loss)
        
        # Average over all positives
        if len(losses) > 0:
            total_loss = torch.stack(losses).mean() * self.pos_weight
        else:
            total_loss = torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        return total_loss


class WeightedRankingLoss(nn.Module):
    """Weighted Ranking Loss with Hard Negative Mining
    
    Class imbalance 문제를 해결하기 위해 positive 샘플에 더 높은 weight 부여
    Hard Negative Mining: positive보다 점수가 높은 negative만 선택하여 loss 계산
    """
    
    def __init__(self, margin=1.0, pos_weight=2.0, auto_weight=True, use_hard_negative=True):
        """
        Args:
            margin: Margin for ranking loss
            pos_weight: Positive sample weight (고정값 사용 시)
            auto_weight: True면 num_neg/num_pos로 자동 계산, False면 pos_weight 사용
            use_hard_negative: True면 hard negative mining 사용
        """
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.auto_weight = auto_weight
        self.use_hard_negative = use_hard_negative
    
    def forward(self, scores, labels):
        """
        scores: (N,) 각 문장의 supporting fact 점수
        labels: (N,) 1.0 for positive, 0.0 for negative
        
        Returns: weighted ranking loss with hard negative mining
        """
        # 크기 불일치 체크
        if scores.shape[0] != labels.shape[0]:
            raise ValueError(f"Size mismatch: scores.shape={scores.shape}, labels.shape={labels.shape}")
        
        pos_mask = labels > 0.5
        neg_mask = labels < 0.5
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            # No positive or negative samples, return 0
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        num_pos = pos_mask.sum().item()
        pos_scores = scores[pos_mask]  # (num_pos,)
        neg_scores_all = scores[neg_mask]  # (num_neg,)
        
        # Hard Negative Mining: positive보다 점수가 높은 negative만 선택
        if self.use_hard_negative and len(pos_scores) > 0:
            # Find negatives that have higher scores than at least one positive
            max_pos_score = pos_scores.max()
            min_pos_score = pos_scores.min()
            mean_pos_score = pos_scores.mean()
            # CRITICAL: 더 엄격한 기준 - 평균보다 높은 negative만 선택
            hard_neg_mask = neg_scores_all > mean_pos_score  # Hard negatives: score > mean(positive scores)
            
            if hard_neg_mask.sum() > 0:
                # Use 50% hard negatives + 50% random negatives for stability
                num_hard = hard_neg_mask.sum().item()
                num_total_neg = len(neg_scores_all)
                
                # Hard negative 비율: 최소 5개, 최대 전체의 50%
                num_hard_to_use = min(num_hard, max(5, num_total_neg // 2))
                num_random = min(num_hard_to_use, num_total_neg - num_hard_to_use)  # 같은 수의 random negatives
                
                # Select hard negatives
                hard_indices = torch.where(hard_neg_mask)[0]
                if len(hard_indices) > num_hard_to_use:
                    # Top-k hard negatives (가장 높은 점수)
                    hard_scores = neg_scores_all[hard_indices]
                    _, top_hard_idx = torch.topk(hard_scores, k=num_hard_to_use)
                    selected_hard = hard_indices[top_hard_idx]
                else:
                    selected_hard = hard_indices
                
                # Select random negatives (excluding hard negatives)
                if num_random > 0:
                    easy_neg_mask = ~hard_neg_mask
                    easy_neg_indices = torch.where(easy_neg_mask)[0]
                    if len(easy_neg_indices) > 0:
                        random_indices = torch.randperm(len(easy_neg_indices))[:num_random]
                        selected_easy = easy_neg_indices[random_indices]
                        # Combine hard and random negatives
                        selected_indices = torch.cat([selected_hard, selected_easy])
                        neg_scores = neg_scores_all[selected_indices]
                    else:
                        neg_scores = neg_scores_all[selected_hard]
                else:
                    neg_scores = neg_scores_all[selected_hard]
            else:
                # No hard negatives found, use all negatives
                neg_scores = neg_scores_all
        else:
            # No hard negative mining, use all negatives
            neg_scores = neg_scores_all
        
        num_neg = len(neg_scores)
        
        if num_neg == 0:
            # No negatives after mining, return small loss
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        # Create pairs: for each positive, compare with selected negatives
        # We want: pos_score > neg_score + margin
        pos_expanded = pos_scores.unsqueeze(1)  # (num_pos, 1)
        neg_expanded = neg_scores.unsqueeze(0)  # (1, num_neg)
        
        # Margin ranking loss: max(0, margin - (pos_score - neg_score))
        # We want: pos_score > neg_score + margin
        losses = F.relu(self.margin - (pos_expanded - neg_expanded))  # (num_pos, num_neg)
        
        # Weight 계산: positive 샘플에 더 높은 weight 부여
        if self.auto_weight:
            # 자동 weight: num_neg / num_pos (class imbalance 반영)
            # 예: pos=2, neg=98 → weight=49.0 (너무 높을 수 있음)
            # 따라서 sqrt를 적용하여 완화
            weight = torch.sqrt(torch.tensor(num_neg / max(num_pos, 1), device=scores.device, dtype=torch.float32))
            # 최대 weight 제한 (너무 높은 weight 방지)
            weight = torch.clamp(weight, min=1.0, max=10.0)
        else:
            weight = self.pos_weight
        
        # Weighted loss: positive 쌍에 더 높은 weight
        weighted_losses = losses * weight  # (num_pos, num_neg)
        
        # CRITICAL: Loss가 0이 되는 것을 방지하기 위해 작은 값 추가
        # 모든 loss가 0이면 gradient가 0이 됨
        # 하지만 이건 잘못된 접근 - loss가 0이면 학습이 잘 된 것
        # 대신, loss가 너무 작으면 gradient가 작아질 수 있으므로 margin을 조정
        
        # Average over all pairs
        loss = weighted_losses.mean()
        
        # CRITICAL: Loss가 0에 가까우면 (모든 positive가 negative보다 높으면) 
        # 여전히 작은 gradient를 유지하기 위해 작은 값 추가
        # 하지만 이건 잘못된 접근이므로 제거
        
        return loss

# Backward compatibility
RankingLoss = WeightedRankingLoss
# Default to ContrastiveLoss for better performance
DefaultLoss = ContrastiveLoss


def train_hierarchical_gat_retriever(train_cache, val_cache=None, num_epochs=20, lr=5e-5):
    """Train Question-aware GAT retriever with Ranking loss"""
    # Find embedding dimension
    sample_emb_dim = None
    for entry in train_cache.values():
        if "node_emb" in entry and entry["node_emb"] is not None and len(entry["node_emb"]) > 0:
            sample_emb_dim = entry["node_emb"].shape[1]
            break
    
    if sample_emb_dim is None:
        raise ValueError("No valid entries for training!")
    
    model = HierarchicalQueryAwareGATRetriever(embedding_dim=sample_emb_dim).to(DEVICE)
    # Learning rate: 1e-4 (더 높은 학습률로 빠른 학습)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # WeightedRankingLoss 사용 (Hard negative mining 활성화)
    loss_fn = WeightedRankingLoss(margin=1.0, auto_weight=True, use_hard_negative=True)
    
    # Learning rate scheduler 추가
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3
    )
    
    print(f"\nTraining Question-aware GAT Retriever for {num_epochs} epochs...")
    
    # Debug: Check cache structure
    skip_reasons = {"no_sents_or_gold": 0, "missing_keys": 0, "empty_emb": 0, "shape_mismatch": 0, "exception": 0}
    
    for ep in range(num_epochs):
        model.train()
        total_loss, n = 0, 0
        for sid, entry in tqdm(train_cache.items(), total=len(train_cache), desc=f"Epoch {ep+1}/{num_epochs}"):
            if len(entry.get("sents", [])) == 0 or len(entry.get("gold", [])) == 0:
                skip_reasons["no_sents_or_gold"] += 1
                continue
            if "adj" not in entry or "node_emb" not in entry or "query_emb" not in entry:
                skip_reasons["missing_keys"] += 1
                continue
            if entry["node_emb"] is None or len(entry["node_emb"]) == 0:
                skip_reasons["empty_emb"] += 1
                continue
            
            try:
                adj = entry["adj"].to(DEVICE)
                node_emb = entry["node_emb"].to(DEVICE)
                query_emb = entry["query_emb"].to(DEVICE)
                
                # Debug shape mismatch
                if adj.shape[0] != node_emb.shape[0]:
                    if n == 0 and ep == 0:  # First sample of first epoch
                        print(f"  WARNING:  Shape mismatch in sample {sid}: adj={adj.shape}, node_emb={node_emb.shape}")
                    skip_reasons["shape_mismatch"] += 1
                    continue
                
                # Hierarchical graph: node_emb는 [query, doc1, ..., docM, sent1, ..., sentN]
                # num_docs 확인
                num_docs = entry.get("num_docs", 0)
                if "node_metadata" in entry:
                    # Count documents from metadata
                    num_docs = sum(1 for meta in entry["node_metadata"] if meta.get("type") == "doc")
                
                # Extract number of sentences from node_emb shape
                total_nodes = node_emb.shape[0]
                n_sentences = total_nodes - 1 - num_docs  # N = total - 1 (query) - M (docs)
                
                # Labels: gold indices are in original sentences (sents), not node_texts
                # Hierarchical graph: node_texts = [query, doc1, ..., docM, sent1, ..., sentN]
                # Build labels (debug disabled)
                labels = build_labels(n_sentences, entry["gold"], entry["node_texts"], entry["sents"], num_docs=num_docs, debug=False)
                
                # Forward pass: hierarchical graph
                scores = model(node_emb, adj, num_docs)
                
                # 크기 불일치 체크
                if scores.shape[0] != labels.shape[0]:
                    if n == 0 and ep == 0:
                        print(f"  WARNING:  Size mismatch in sample {sid}: scores={scores.shape}, labels={labels.shape}, n_sentences={n_sentences}")
                    skip_reasons["shape_mismatch"] += 1
                    continue
                
                loss = loss_fn(scores, labels)
                
                opt.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                opt.step()
                total_loss += loss.item()
                n += 1
            except Exception as e:
                if n == 0 and ep == 0:  # First error of first epoch
                    print(f"  WARNING:  Exception in sample {sid}: {e}")
                    import traceback
                    traceback.print_exc()
                skip_reasons["exception"] += 1
                continue
        
        avg_loss = total_loss / max(1, n)
        
        # Validation loss 계산 (val_cache가 제공된 경우)
        val_loss = None
        if val_cache is not None:
            model.eval()
            val_total_loss, val_n = 0, 0
            with torch.no_grad():
                for sid, entry in val_cache.items():
                    if len(entry.get("sents", [])) == 0 or len(entry.get("gold", [])) == 0:
                        continue
                    if "adj" not in entry or "node_emb" not in entry or "query_emb" not in entry:
                        continue
                    if entry["node_emb"] is None or len(entry["node_emb"]) == 0:
                        continue
                    
                    try:
                        adj = entry["adj"].to(DEVICE)
                        node_emb = entry["node_emb"].to(DEVICE)
                        query_emb = entry["query_emb"].to(DEVICE)
                        
                        if adj.shape[0] != node_emb.shape[0]:
                            continue
                        
                        num_docs = entry.get("num_docs", 0)
                        if "node_metadata" in entry:
                            num_docs = sum(1 for meta in entry["node_metadata"] if meta.get("type") == "doc")
                        
                        total_nodes = node_emb.shape[0]
                        n_sentences = total_nodes - 1 - num_docs
                        
                        labels = build_labels(n_sentences, entry["gold"], entry["node_texts"], entry["sents"], num_docs=num_docs, debug=False)
                        
                        scores = model(node_emb, adj, num_docs)
                        
                        if scores.shape[0] != labels.shape[0]:
                            continue
                        
                        loss = loss_fn(scores, labels)
                        val_total_loss += loss.item()
                        val_n += 1
                    except Exception:
                        continue
            
            if val_n > 0:
                val_loss = val_total_loss / val_n
        
        # Print epoch results
        if val_loss is not None:
            print(f"Epoch {ep+1}/{num_epochs} - Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {ep+1}/{num_epochs} - Train Loss: {avg_loss:.6f}")
        
        # Debug: Print skip reasons for first epoch
        if ep == 0 and n == 0:
            print(f"  WARNING:  No samples processed! Skip reasons:")
            for reason, count in skip_reasons.items():
                if count > 0:
                    print(f"    - {reason}: {count}")
        
        # Learning rate scheduling (use val_loss if available, otherwise train_loss)
        scheduler.step(val_loss if val_loss is not None else avg_loss)
    
    return model
from .config import DEVICE, AVAILABLE_GPUS, MODEL_NAME, HIDDEN_DIM, GNN_LAYERS, THRESHOLD

# ==================== Retrieval ====================
def retrieve_supporting_facts(entry, model, threshold=THRESHOLD, top_k=None, gold_count=None):
    """Retrieve supporting facts using Hierarchical Query-Aware GAT
    
    Args:
        entry: Cache entry with graph data
        model: Trained retriever model
        threshold: Probability threshold for filtering (default: THRESHOLD)
        top_k: Maximum number of facts to retrieve (if None, uses gold_count or 3)
        gold_count: Number of gold facts (used to determine top_k if provided)
    
    Returns:
        List of dicts with keys: 'text', 'doc_id', 'local_sent_id'
        For backward compatibility, also returns List[str] if only text is needed
    """
    eval_model = model.module if hasattr(model, 'module') else model
    eval_model.eval()
    
    adj = entry["adj"].to(DEVICE)
    node_emb = entry["node_emb"].to(DEVICE)
    node_texts = entry["node_texts"]  # [query, doc1, ..., docM, sent1, ..., sentN]
    sents = entry["sents"]  # Original sentences (for matching)
    sent_metadata = entry.get("sent_metadata", [])  # (doc_id, local_sent_id) for each sentence
    node_metadata = entry.get("node_metadata", [])  # Full node metadata
    
    # Get num_docs
    num_docs = entry.get("num_docs", 0)
    if "node_metadata" in entry:
        num_docs = sum(1 for meta in entry["node_metadata"] if meta.get("type") == "doc")
    
    with torch.no_grad():
        # Forward pass: hierarchical graph
        logits = eval_model(node_emb, adj, num_docs)  # Returns (n_sentences,) scores
        probs = torch.sigmoid(logits)  # Sentence probabilities
    
    # Determine top_k: prioritize gold_count, then top_k, then default to 3
    if top_k is None:
        if gold_count is not None:
            top_k = max(2, min(gold_count, 10))  # At least 2, at most 10, prefer gold_count
        else:
            top_k = 3  # Default
    
    # Get top-k sentences
    k = min(top_k, len(probs))
    topk_probs, topk_indices = torch.topk(probs, k=k)
    
    # Adaptive threshold: if all scores are very low, use a lower threshold
    max_prob = probs.max().item()
    mean_prob = probs.mean().item()
    
    # If max probability is below threshold, use adaptive threshold
    adaptive_threshold = threshold
    if max_prob < threshold:
        # Use a threshold based on top-k scores: take top-k with at least 0.1 probability
        adaptive_threshold = max(0.1, min(topk_probs[-1].item() if len(topk_probs) > 0 else 0.1, threshold * 0.7))
        if not hasattr(retrieve_supporting_facts, '_threshold_warning_shown'):
            retrieve_supporting_facts._threshold_warning_shown = False
        if not retrieve_supporting_facts._threshold_warning_shown:
            print(f"\n[Retriever Warning] Max probability ({max_prob:.4f}) is below threshold ({threshold:.4f})")
            print(f"  Using adaptive threshold: {adaptive_threshold:.4f}")
            retrieve_supporting_facts._threshold_warning_shown = True
    
    # Debug: Print score statistics for first few samples (DISABLED)
    # if not hasattr(retrieve_supporting_facts, '_debug_count'):
    #     retrieve_supporting_facts._debug_count = 0
    # retrieve_supporting_facts._debug_count += 1
    # 
    # if retrieve_supporting_facts._debug_count <= 3:
    #     print(f"\n[Retriever Debug] Sample {retrieve_supporting_facts._debug_count}")
    #     print(f"  Score stats: min={probs.min().item():.4f}, max={probs.max().item():.4f}, mean={probs.mean().item():.4f}, std={probs.std().item():.4f}")
    #     print(f"  Top-{k} scores: {topk_probs.cpu().tolist()}")
    #     print(f"  Original threshold: {threshold}, Adaptive threshold: {adaptive_threshold:.4f}")
    #     print(f"  Gold count: {gold_count}, Top-k: {top_k}")
    
    pred_facts = []  # List of dicts: {'text': str, 'doc_id': int, 'local_sent_id': int}
    seen_sents = set()
    
    # CRITICAL: Threshold 제거, Top-k만 사용 (모델 점수가 낮아서 threshold가 의미 없음)
    # Strategy 1 제거: Threshold-based selection은 점수가 낮아서 효과 없음
    
    # Strategy 2: Top-k 사용 (항상 사용)
    for idx in topk_indices.cpu().tolist():
        if len(pred_facts) >= top_k:
            break
        sent_node_idx = 1 + num_docs + idx
        if sent_node_idx < len(node_texts):
            sent_text = node_texts[sent_node_idx]
            if sent_text in sents and sent_text not in seen_sents:
                # Get (doc_id, local_sent_id)
                doc_id, local_sent_id = -1, -1
                if sent_node_idx < len(node_metadata):
                    meta = node_metadata[sent_node_idx]
                    if meta.get("type") == "sentence":
                        doc_id = meta.get("doc_id", -1)
                        local_sent_id = meta.get("local_sent_id", -1)
                elif idx < len(sent_metadata):
                    doc_id, local_sent_id = sent_metadata[idx]
                
                pred_facts.append({
                    'text': sent_text,
                    'doc_id': doc_id,
                    'local_sent_id': local_sent_id
                })
                seen_sents.add(sent_text)
    
    # Strategy 3: Final fallback if still empty
    if not pred_facts:
        for i in range(min(top_k, len(sents))):
            doc_id, local_sent_id = sent_metadata[i] if i < len(sent_metadata) else (-1, -1)
            pred_facts.append({
                'text': sents[i],
                'doc_id': doc_id,
                'local_sent_id': local_sent_id
            })
    
    return pred_facts[:top_k]  # Return up to top_k facts with metadata

