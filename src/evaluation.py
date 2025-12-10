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
from .retriever import retrieve_supporting_facts
from .generator import compute_answer_em_f1

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==================== Evaluation ====================
def evaluate_hierarchical_graph_rag_pipeline(cache, retriever_model, generator_model, generator_tokenizer, mode="joint"):
    """
    Evaluate full pipeline
    mode: 'sup' = retrieval only, 'answer' = gold→answer, 'joint' = retrieved→answer
    """
    eval_model = retriever_model.module if hasattr(retriever_model, 'module') else retriever_model
    eval_model.eval()
    generator_model.eval()
    
    f1s_sup, ems_sup = [], []
    f1s_ans, ems_ans = [], []
    hit_at_1, hit_at_3, hit_at_5, hit_at_10 = [], [], [], []
    
    error_count = 0
    skipped_count = 0
    processed_count = 0
    debug_count = 0  # Track number of debug outputs
    
    # Debug: Cache size (removed for cleaner output)
    
    for sid, entry in tqdm(cache.items(), desc=f"Evaluating {mode}"):
        gold_idx = entry.get("gold", [])
        if not gold_idx or len(entry.get("sents", [])) == 0:
            skipped_count += 1
            continue
        
        try:
            # Retrieval evaluation (only for 'sup' mode or when needed for 'joint')
            pred_facts = None  # Initialize for use in joint mode
            if mode == "sup" or mode == "joint":
                gold_sents = [entry["sents"][i] for i in gold_idx if i < len(entry["sents"])]
                if not gold_sents:
                    skipped_count += 1
                    continue
                
                # For Hit@K calculation, retrieve top-10
                pred_facts_top10 = retrieve_supporting_facts(
                    entry, 
                    eval_model, 
                    threshold=THRESHOLD,
                    top_k=10
                )
                
                # For EM/F1, use gold_count to match the number
                gold_count_for_retrieval = max(2, min(len(gold_sents), 10))  # At least 2, at most 10
                pred_facts = retrieve_supporting_facts(
                    entry, 
                    eval_model, 
                    threshold=THRESHOLD,
                    gold_count=gold_count_for_retrieval
                )
                
                # Debug output removed for cleaner logs
                debug_count += 1
                
                if not pred_facts or len(pred_facts) == 0:
                    # Try fallback: get top-k without filtering
                    if not pred_facts_top10 or len(pred_facts_top10) == 0:
                        if debug_count <= 3:
                            print(f"  WARNING: Both pred_facts and pred_facts_top10 are empty!")
                        skipped_count += 1
                        continue
                    else:
                        # Use top10 if pred_facts is empty
                        pred_facts = pred_facts_top10[:len(gold_sents)] if gold_sents else pred_facts_top10[:3]
                if not pred_facts_top10 or len(pred_facts_top10) == 0:
                    # Use pred_facts as fallback for top10
                    pred_facts_top10 = pred_facts
                
                # Extract text from dict format (backward compatibility)
                pred_sents = [fact['text'] if isinstance(fact, dict) else fact for fact in pred_facts]
                pred_sents_top10 = [fact['text'] if isinstance(fact, dict) else fact for fact in pred_facts_top10]
                
                f1_sup = supporting_fact_f1(pred_sents, gold_sents)
                em_sup = supporting_fact_em(pred_sents, gold_sents)
                f1s_sup.append(f1_sup)
                ems_sup.append(em_sup)
                
                # Compute Hit@K
                from utils.graph_utils import normalize_string
                gold_normalized = set([normalize_string(g) for g in gold_sents])
                
                def compute_hit_at_k(retrieved_sents, gold_set, k):
                    if not retrieved_sents or not gold_set:
                        return 0.0
                    top_k_retrieved = retrieved_sents[:k]
                    retrieved_normalized = set([normalize_string(s) for s in top_k_retrieved])
                    return 1.0 if (retrieved_normalized & gold_set) else 0.0
                
                hit_at_1.append(compute_hit_at_k(pred_sents_top10, gold_normalized, 1))
                hit_at_3.append(compute_hit_at_k(pred_sents_top10, gold_normalized, 3))
                hit_at_5.append(compute_hit_at_k(pred_sents_top10, gold_normalized, 5))
                hit_at_10.append(compute_hit_at_k(pred_sents_top10, gold_normalized, 10))
                
                # Debug output for first 3 samples (retriever mode only) - DISABLED
                # if mode == "sup" and debug_count < 3:
                #     debug_count += 1
                #     print(f"\n{'='*80}")
                #     print(f"[DEBUG] Sample {sid} - Retriever Results")
                #     print(f"{'='*80}")
                #     
                #     # Question
                #     question = entry.get("question", "N/A")
                #     print(f"\nQuestion: {question}")
                #     
                #     # Gold supporting facts
                #     print(f"\n[GOLD] Supporting Facts ({len(gold_sents)}):")
                #     sent_metadata = entry.get("sent_metadata", [])
                #     for i, gi in enumerate(gold_idx):
                #         if gi < len(entry.get("sents", [])):
                #             gold_sent = entry["sents"][gi]
                #             doc_id, local_sent_id = sent_metadata[gi] if gi < len(sent_metadata) else (-1, -1)
                #             print(f"  [{i+1}] (doc_id={doc_id}, local_sent_id={local_sent_id})")
                #             print(f"      \"{gold_sent[:100]}{'...' if len(gold_sent) > 100 else ''}\"")
                #     
                #     # Predicted supporting facts
                #     print(f"\n[PREDICTED] Supporting Facts ({len(pred_facts)}):")
                #     for i, fact in enumerate(pred_facts):
                #         if isinstance(fact, dict):
                #             text = fact.get('text', '')
                #             doc_id = fact.get('doc_id', -1)
                #             local_sent_id = fact.get('local_sent_id', -1)
                #             print(f"  [{i+1}] (doc_id={doc_id}, local_sent_id={local_sent_id})")
                #             print(f"      \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
                #         else:
                #             print(f"  [{i+1}] \"{str(fact)[:100]}{'...' if len(str(fact)) > 100 else ''}\"")
                #     
                #     # Score details
                #     print(f"\n[SCORES]")
                #     print(f"  EM: {em_sup:.4f} {'OK:' if em_sup == 1.0 else 'ERROR:'}")
                #     print(f"  F1: {f1_sup:.4f}")
                #     
                #     # Match analysis
                #     gold_set = set(gold_sents)
                #     pred_set = set(pred_sents)
                #     matched = gold_set & pred_set
                #     gold_only = gold_set - pred_set
                #     pred_only = pred_set - gold_set
                #     
                #     print(f"\n[MATCH ANALYSIS]")
                #     print(f"  Matched: {len(matched)}/{len(gold_set)}")
                #     if matched:
                #         print(f"    OK: Correctly retrieved:")
                #         for sent in matched:
                #             print(f"      - \"{sent[:80]}{'...' if len(sent) > 80 else ''}\"")
                #     if gold_only:
                #         print(f"    ERROR: Missed (in gold but not retrieved):")
                #         for sent in list(gold_only)[:3]:  # Show max 3
                #             print(f"      - \"{sent[:80]}{'...' if len(sent) > 80 else ''}\"")
                #     if pred_only:
                #         print(f"    WARNING:  Extra (retrieved but not in gold):")
                #         for sent in list(pred_only)[:3]:  # Show max 3
                #             print(f"      - \"{sent[:80]}{'...' if len(sent) > 80 else ''}\"")
                #     
                #     print(f"{'='*80}\n")
            else:
                # For 'answer' mode, use gold sentences directly
                gold_sents = [entry["sents"][i] for i in gold_idx if i < len(entry["sents"])]
                pred_sents = gold_sents  # Not used for evaluation, just for consistency
            
            # Answer generation (only for 'answer' or 'joint' mode)
            if mode in ["answer", "joint"]:
                question = entry["question"]
                answer = entry["answer"]
                
                if mode == "answer":
                    # Use GOLD supporting facts (all of them, no limit)
                    evidences = gold_sents
                    max_evidences = len(gold_sents)  # Use all gold facts
                else:  # joint
                    # Use RETRIEVED supporting facts (max 3)
                    # Extract text from dict format
                    if pred_facts and isinstance(pred_facts[0], dict):
                        evidences = [fact['text'] for fact in pred_facts]
                    else:
                        evidences = pred_sents  # Already extracted above
                    max_evidences = 3  # Limit to 3 for retrieved facts
                
                if not evidences:
                    skipped_count += 1
                    continue
                
                if question and answer:
                    # T5: Concatenate question and evidences into single string
                    input_parts = [f"question: {question}"]
                    for evidence in evidences[:max_evidences]:
                        if evidence:
                            input_parts.append(f"context: {evidence}")
                    input_text = " ".join(input_parts)
                    
                    # Generate with T5
                    inputs = generator_tokenizer(
                        input_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=512
                    ).to(DEVICE)
                    
                    # Get model (handle both wrapped and unwrapped models for compatibility)
                    model_to_use = generator_model.base_model if hasattr(generator_model, 'base_model') else generator_model
                    
                    with torch.no_grad():
                        outputs = model_to_use.generate(
                            **inputs,
                            max_new_tokens=20,
                            num_beams=1,
                            do_sample=False
                        )
                    pred_answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # 후처리
                    pred_answer = pred_answer.strip()
                    patterns_to_remove = [
                        r'^the answer is\s+',
                        r'^answer:\s*',
                        r'\s+is the answer\.?$',
                        r'^answer\s+is\s+',
                    ]
                    for pattern in patterns_to_remove:
                        pred_answer = re.sub(pattern, '', pred_answer, flags=re.IGNORECASE)
                    pred_answer = pred_answer.strip()
                    
                    if not pred_answer or pred_answer.strip() == "":
                        pred_answer = "unknown"
                    
                    em_ans, f1_ans = compute_answer_em_f1(pred_answer, answer)
                    f1s_ans.append(f1_ans)
                    ems_ans.append(em_ans)
                    processed_count += 1
        except Exception as e:
            error_count += 1
            continue
    
    # Return results based on mode
    result = {}
    if mode == "sup":
        # Retriever only: return sup metrics + Hit@K
        sup_em_mean = np.mean(ems_sup) if ems_sup else 0.0
        sup_f1_mean = np.mean(f1s_sup) if f1s_sup else 0.0
        hit_at_1_mean = np.mean(hit_at_1) if hit_at_1 else 0.0
        hit_at_3_mean = np.mean(hit_at_3) if hit_at_3 else 0.0
        hit_at_5_mean = np.mean(hit_at_5) if hit_at_5 else 0.0
        hit_at_10_mean = np.mean(hit_at_10) if hit_at_10 else 0.0
        result = {
            "sup_em": sup_em_mean, 
            "sup_f1": sup_f1_mean,
            "hit@1": hit_at_1_mean,
            "hit@3": hit_at_3_mean,
            "hit@5": hit_at_5_mean,
            "hit@10": hit_at_10_mean
        }
    elif mode in ["answer", "joint"]:
        # Generator/Joint: return ans metrics only
        ans_em_mean = np.mean(ems_ans) if ems_ans and len(ems_ans) > 0 else 0.0
        ans_f1_mean = np.mean(f1s_ans) if f1s_ans and len(f1s_ans) > 0 else 0.0
        result = {"ans_em": ans_em_mean, "ans_f1": ans_f1_mean}
    
    return result

