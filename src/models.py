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

# Add parent directory to path for utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.graph_utils import flatten_context, get_gold_indices, supporting_fact_em, supporting_fact_f1, batch_embed, build_hierarchical_graph
from utils.gpu_utils import set_device_auto, get_available_gpus
from .config import DEVICE, AVAILABLE_GPUS, MODEL_NAME, HIDDEN_DIM, GNN_LAYERS, THRESHOLD

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==================== Question-aware GAT Model ====================
class GATLayer(nn.Module):
    """Graph Attention Layer"""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        # Linear transformations for each head
        self.W = nn.ModuleList([nn.Linear(in_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1)))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for w in self.W:
            nn.init.xavier_uniform_(w.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, h, adj):
        """
        h: (N, in_dim) node features
        adj: (N, N) adjacency matrix
        """
        N = h.size(0)
        adj_N = adj.size(0)
        
        # 크기 불일치 체크 및 수정
        if N != adj_N:
            # adj 크기에 맞춰 h를 자르거나, h 크기에 맞춰 adj를 확장
            if N > adj_N:
                # h가 더 크면 adj에 맞춰 자름
                h = h[:adj_N]
                N = adj_N
            else:
                # adj가 더 크면 h에 맞춰 adj를 자름
                adj = adj[:N, :N]
        
        # Multi-head attention
        head_outputs = []
        for head in range(self.num_heads):
            Wh = self.W[head](h)  # (N, head_dim)
            
            # Compute attention scores
            h_expanded_i = Wh.unsqueeze(1).expand(-1, N, -1)  # (N, N, head_dim)
            h_expanded_j = Wh.unsqueeze(0).expand(N, -1, -1)  # (N, N, head_dim)
            
            # Concatenate and compute attention
            h_concat = torch.cat([h_expanded_i, h_expanded_j], dim=2)  # (N, N, 2*head_dim)
            e_ij = (h_concat @ self.a).squeeze(-1)  # (N, N)
            e_ij = self.leaky_relu(e_ij)
            
            # Apply adjacency mask (크기 확인 후 적용)
            if e_ij.shape == adj.shape:
                e_ij = e_ij.masked_fill(adj == 0, float('-inf'))
            else:
                # 크기가 다르면 adj를 e_ij 크기에 맞춤
                if adj.size(0) < e_ij.size(0):
                    # adj를 확장 (padding with zeros)
                    pad_size = e_ij.size(0) - adj.size(0)
                    adj_padded = F.pad(adj, (0, pad_size, 0, pad_size), value=0)
                    e_ij = e_ij.masked_fill(adj_padded == 0, float('-inf'))
                else:
                    # adj를 자름
                    adj = adj[:e_ij.size(0), :e_ij.size(0)]
                    e_ij = e_ij.masked_fill(adj == 0, float('-inf'))
            
            # Softmax attention
            attention = F.softmax(e_ij, dim=1)  # (N, N)
            attention = self.dropout(attention)
            
            # Aggregate
            h_new = torch.matmul(attention, Wh)  # (N, head_dim)
            head_outputs.append(h_new)
        
        # Concatenate all heads
        h_out = torch.cat(head_outputs, dim=1)  # (N, out_dim)
        
        return h_out


class HierarchicalQueryAwareGATRetriever(nn.Module):
    """Hierarchical Query-Aware GAT for sentence retrieval
    
    구조:
    - Nodes: [query, doc1, ..., docM, sent1, ..., sentN]
    - GAT: Query, documents, sentences 모두 포함하여 reasoning
    - Query-guided message passing: Query → Doc → Sent, Query → Sent → Sent
    """
    
    def __init__(self, embedding_dim=384, hidden_dim=HIDDEN_DIM, num_layers=GNN_LAYERS, num_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Projection for all nodes (query, docs, sents)
        self.proj = nn.Linear(embedding_dim, hidden_dim)
        
        # GAT layers (all nodes included)
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(GATLayer(hidden_dim, hidden_dim, num_heads))
        
        # Layer normalization to prevent over-smoothing
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Dropout for regularization (increased to prevent over-smoothing)
        self.dropout = nn.Dropout(0.2)
        
        # Query fusion layer: for combining sentence + query + similarity + attention
        # Input: (hidden_dim * 2 + 2) = [sent_h, query_weighted, cosine_sim, attention_feature]
        self.query_fusion = nn.Linear(hidden_dim * 2 + 2, hidden_dim)
        
        # Output layer: sentence relevance scores
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Improved initialization
        self._init_weights()
    
    def _init_weights(self):
        """Improved weight initialization"""
        # Xavier initialization for projection
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
        # Query fusion layer initialization
        nn.init.xavier_uniform_(self.query_fusion.weight)
        nn.init.zeros_(self.query_fusion.bias)
        
        # Output layer: small positive bias to encourage learning
        # 더 강한 initialization으로 positive 점수를 높이기
        nn.init.xavier_uniform_(self.output_layer.weight, gain=2.0)  # Higher gain (1.5 → 2.0)
        self.output_layer.bias.data.fill_(0.5)  # Higher bias to push scores up (0.2 → 0.5)
    
    def forward(self, node_embeddings, adj, num_docs):
        """
        node_embeddings: (1+M+N, embedding_dim)
          - [0]: query
          - [1:1+M]: documents
          - [1+M:1+M+N]: sentences
        
        adj: (1+M+N, 1+M+N) - hierarchical graph with query
        
        num_docs: M (number of documents)
        
        Returns:
            scores: (N,) - 각 문장의 supporting fact 확률 (logits)
        """
        total_nodes = node_embeddings.size(0)
        M = num_docs
        N = total_nodes - 1 - M  # Number of sentences
        
        if N == 0:
            return torch.tensor([], device=node_embeddings.device)
        
        # Project all nodes
        h = self.proj(node_embeddings)  # (1+M+N, hidden_dim)
        h = F.relu(h)
        h = self.dropout(h)
        
        # CRITICAL: Query 정보를 sentence에 직접 주입 (GAT 전에)
        # 이렇게 하면 GAT가 query 정보를 활용하여 message passing 가능
        query_emb = h[0:1]  # (1, hidden_dim) - query embedding
        sent_emb_start = 1 + M
        sent_emb = h[sent_emb_start:]  # (N, hidden_dim) - sentence embeddings
        
        # Query-sentence similarity를 계산하여 query 정보를 주입
        query_norm = query_emb / (torch.norm(query_emb, dim=1, keepdim=True) + 1e-8)  # (1, hidden_dim)
        sent_norm = sent_emb / (torch.norm(sent_emb, dim=1, keepdim=True) + 1e-8)  # (N, hidden_dim)
        cosine_sim = torch.mm(query_norm, sent_norm.t()).squeeze(0)  # (N,) - 각 sentence와의 cosine similarity
        cosine_sim = cosine_sim.unsqueeze(1)  # (N, 1)
        
        # Query 정보를 sentence에 직접 더하기 (cosine similarity로 가중치 적용)
        query_expanded = query_emb.expand(N, -1)  # (N, hidden_dim)
        # Cosine similarity가 높을수록 더 많은 query 정보를 받음
        # CRITICAL: 더 강한 query 정보 주입 (0.5 → 0.8)
        sent_with_query = sent_emb + 0.8 * query_expanded * cosine_sim  # (N, hidden_dim)
        
        # FIX: In-place operation 대신 새로운 tensor 생성 (gradient 계산을 위해)
        h = torch.cat([
            h[0:1],  # Query
            h[1:sent_emb_start],  # Documents
            sent_with_query  # Sentences with query information
        ], dim=0)
        
        # GAT layers (query, docs, sents 모두 포함하여 reasoning)
        # Residual connection 비율 조정: 0.5 * new + 0.5 * residual (더 균형잡힌 학습)
        for i, gat_layer in enumerate(self.gat_layers):
            h_residual = h  # Residual connection
            # Layer normalization BEFORE GAT to prevent over-smoothing
            h = self.layer_norms[i](h)
            h = gat_layer(h, adj)
            h = F.relu(h)
            h = self.dropout(h)
            # Balanced residual connection: 0.5 * new + 0.5 * residual
            h = 0.5 * h + 0.5 * h_residual
        
        # Extract sentence and query representations
        query_h = h[0:1]  # (1, hidden_dim) - query representation
        sent_h = h[1+M:]  # (N, hidden_dim) - sentence representations
        
        # Cross-attention: Query가 sentences에 attention
        # Query (1, hidden_dim) as query, Sentences (N, hidden_dim) as key and value
        # Compute attention scores: query attends to each sentence
        # Attention: Q @ K^T / sqrt(d_k)
        # query_h: (1, hidden_dim), sent_h: (N, hidden_dim)
        attention_scores = torch.mm(query_h, sent_h.t()) / (self.hidden_dim ** 0.5)  # (1, N)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (1, N) - query attends to all sentences
        
        # CRITICAL FIX: 각 sentence마다 다른 query attention을 적용
        # attention_weights: (1, N) - query가 각 sentence에 주는 attention
        attention_weights_expanded = attention_weights.t()  # (N, 1) - 각 sentence의 attention weight
        
        # 방법 1: Attention-weighted query information
        # 높은 attention을 받는 sentence는 더 많은 query 정보를 받음
        # CRITICAL: 더 강한 query 정보 반영 (attention weight에 offset 추가)
        query_h_expanded = query_h.expand(N, -1)  # (N, hidden_dim)
        query_weighted = query_h_expanded * (attention_weights_expanded + 0.5)  # (N, hidden_dim) - 더 강한 query 정보
        
        # 방법 2: Query-sentence cosine similarity를 직접 계산 (이미 GAT 전에 계산했지만 다시 계산)
        query_norm_final = query_h / (torch.norm(query_h, dim=1, keepdim=True) + 1e-8)  # (1, hidden_dim)
        sent_norm_final = sent_h / (torch.norm(sent_h, dim=1, keepdim=True) + 1e-8)  # (N, hidden_dim)
        cosine_sim_final = torch.mm(query_norm_final, sent_norm_final.t()).squeeze(0)  # (N,) - 각 sentence와의 cosine similarity
        cosine_sim_final = cosine_sim_final.unsqueeze(1)  # (N, 1)
        
        # 방법 3: Attention weight 자체를 feature로 사용 (query가 sentence에 얼마나 관심있는지)
        attention_feature = attention_weights.t()  # (N, 1)
        
        # Combine: original sentence + weighted query + cosine similarity + attention weight
        query_guided_h = torch.cat([
            sent_h,  # Original sentence representation (이미 query 정보가 주입됨)
            query_weighted,  # Query information weighted by attention
            cosine_sim_final,  # Direct query-sentence similarity
            attention_feature  # Attention weight as feature
        ], dim=1)  # (N, hidden_dim * 2 + 2)
        
        # Project back to hidden_dim
        query_guided_h = self.query_fusion(query_guided_h)  # (N, hidden_dim)
        query_guided_h = F.relu(query_guided_h)
        query_guided_h = self.dropout(query_guided_h)
        
        # Final scores
        sentence_scores = self.output_layer(query_guided_h).squeeze(-1)  # (N,)
        
        # CRITICAL: Cosine similarity를 score에 직접 더하기 (query와의 유사도 반영)
        # 이렇게 하면 모델이 query와의 유사도를 직접 활용하여 positive 점수 상승
        cosine_sim_for_score = cosine_sim_final.squeeze(1)  # (N,) - cosine similarity를 score에 직접 반영
        sentence_scores = sentence_scores + 0.5 * cosine_sim_for_score  # Cosine similarity를 직접 더함
        
        return sentence_scores


# Backward compatibility: Keep old class name
QuestionAwareGATRetriever = HierarchicalQueryAwareGATRetriever

