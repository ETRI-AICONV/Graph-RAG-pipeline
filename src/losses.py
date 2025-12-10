"""
Loss functions for HotpotQA Graph-RAG Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import DEVICE

class ContrastiveLoss(nn.Module):
    """InfoNCE-based Contrastive Loss for Retrieval
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
        pos_expanded = pos_scores.unsqueeze(1)  # (num_pos, 1)
        neg_expanded = neg_scores.unsqueeze(0)  # (1, num_neg)
        
        # Margin ranking loss: max(0, margin - (pos_score - neg_score))
        losses = F.relu(self.margin - (pos_expanded - neg_expanded))  # (num_pos, num_neg)
        
        # Weight 계산: positive 샘플에 더 높은 weight 부여
        if self.auto_weight:
            weight = torch.sqrt(torch.tensor(num_neg / max(num_pos, 1), device=scores.device, dtype=torch.float32))
            # 최대 weight 제한 (너무 높은 weight 방지)
            weight = torch.clamp(weight, min=1.0, max=10.0)
        else:
            weight = self.pos_weight
        
        weighted_losses = losses * weight  # (num_pos, num_neg)
        
        # Average over all pairs
        loss = weighted_losses.mean()
        
        return loss

# Backward compatibility
RankingLoss = WeightedRankingLoss
DefaultLoss = ContrastiveLoss
