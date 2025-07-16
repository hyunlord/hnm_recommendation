"""H&M 추천 시스템을 위한 평가 지표."""

import torch
import numpy as np
from typing import List, Dict, Union
from torchmetrics import Metric
import torch.nn.functional as F


class MeanAveragePrecision(Metric):
    """Mean Average Precision @ K 지표."""
    
    def __init__(self, k: int = 12):
        super().__init__()
        self.k = k
        self.add_state("ap_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """지표 업데이트.
        
        Args:
            preds: 예측된 아이템 [batch_size, n_items] 또는 순위
            target: 실제 아이템 [batch_size, n_true_items]
            mask: target의 유효한 아이템을 위한 마스크 [batch_size, n_true_items]
        """
        batch_size = preds.shape[0]
        
        for i in range(batch_size):
            # 이 사용자에 대한 예측 가져오기
            if len(preds.shape) == 2:
                # preds가 점수인 경우, top-k를 가져와야 함
                _, pred_items = torch.topk(preds[i], k=min(self.k, preds.shape[1]))
            else:
                # preds가 이미 아이템 인덱스인 경우
                pred_items = preds[i][:self.k]
            
            # 이 사용자의 실제 아이템 가져오기
            if mask is not None:
                true_items = target[i][mask[i]]
            else:
                true_items = target[i]
            
            # 평균 정밀도 계산
            ap = self._average_precision(pred_items, true_items)
            self.ap_sum += ap
            self.count += 1
    
    def _average_precision(self, pred_items: torch.Tensor, true_items: torch.Tensor) -> torch.Tensor:
        """한 사용자에 대한 평균 정밀도 계산."""
        if len(true_items) == 0:
            return torch.tensor(0.0)
        
        score = 0.0
        num_hits = 0.0
        
        for i, pred_item in enumerate(pred_items):
            if pred_item in true_items:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / min(len(true_items), self.k)
    
    def compute(self):
        """최종 지표 계산."""
        return self.ap_sum / self.count if self.count > 0 else torch.tensor(0.0)


class RecallAtK(Metric):
    """Recall @ K 지표."""
    
    def __init__(self, k: int = 12):
        super().__init__()
        self.k = k
        self.add_state("recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """지표 업데이트."""
        batch_size = preds.shape[0]
        
        for i in range(batch_size):
            # 예측 가져오기
            if len(preds.shape) == 2:
                _, pred_items = torch.topk(preds[i], k=min(self.k, preds.shape[1]))
            else:
                pred_items = preds[i][:self.k]
            
            # 실제 아이템 가져오기
            if mask is not None:
                true_items = target[i][mask[i]]
            else:
                true_items = target[i]
            
            if len(true_items) > 0:
                # 적중 횟수 계산
                hits = sum(1 for item in pred_items if item in true_items)
                recall = hits / len(true_items)
                self.recall_sum += recall
                self.count += 1
    
    def compute(self):
        """최종 지표 계산."""
        return self.recall_sum / self.count if self.count > 0 else torch.tensor(0.0)


class PrecisionAtK(Metric):
    """Precision @ K 지표."""
    
    def __init__(self, k: int = 12):
        super().__init__()
        self.k = k
        self.add_state("precision_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """지표 업데이트."""
        batch_size = preds.shape[0]
        
        for i in range(batch_size):
            # 예측 가져오기
            if len(preds.shape) == 2:
                _, pred_items = torch.topk(preds[i], k=min(self.k, preds.shape[1]))
            else:
                pred_items = preds[i][:self.k]
            
            # 실제 아이템 가져오기
            if mask is not None:
                true_items = target[i][mask[i]]
            else:
                true_items = target[i]
            
            # 적중 횟수 계산
            hits = sum(1 for item in pred_items if item in true_items)
            precision = hits / len(pred_items) if len(pred_items) > 0 else 0
            self.precision_sum += precision
            self.count += 1
    
    def compute(self):
        """최종 지표 계산."""
        return self.precision_sum / self.count if self.count > 0 else torch.tensor(0.0)


class NDCGAtK(Metric):
    """Normalized Discounted Cumulative Gain @ K 지표."""
    
    def __init__(self, k: int = 12):
        super().__init__()
        self.k = k
        self.add_state("ndcg_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """지표 업데이트."""
        batch_size = preds.shape[0]
        
        for i in range(batch_size):
            # 예측 가져오기
            if len(preds.shape) == 2:
                _, pred_items = torch.topk(preds[i], k=min(self.k, preds.shape[1]))
            else:
                pred_items = preds[i][:self.k]
            
            # 실제 아이템 가져오기
            if mask is not None:
                true_items = target[i][mask[i]]
            else:
                true_items = target[i]
            
            if len(true_items) > 0:
                # NDCG 계산
                ndcg = self._ndcg(pred_items, true_items)
                self.ndcg_sum += ndcg
                self.count += 1
    
    def _ndcg(self, pred_items: torch.Tensor, true_items: torch.Tensor) -> torch.Tensor:
        """한 사용자에 대한 NDCG 계산."""
        dcg = 0.0
        for i, pred_item in enumerate(pred_items):
            if pred_item in true_items:
                dcg += 1.0 / np.log2(i + 2)
        
        # 이상적인 DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), self.k)))
        
        return dcg / idcg if idcg > 0 else torch.tensor(0.0)
    
    def compute(self):
        """최종 지표 계산."""
        return self.ndcg_sum / self.count if self.count > 0 else torch.tensor(0.0)


def evaluate_recommendations(
    predictions: Dict[int, List[int]], 
    ground_truth: Dict[int, List[int]], 
    k: int = 12
) -> Dict[str, float]:
    """여러 지표를 사용하여 추천 평가.
    
    Args:
        predictions: user_id를 예측된 아이템 리스트로 매핑하는 Dict
        ground_truth: user_id를 실제 아이템 리스트로 매핑하는 Dict
        k: 지표의 컷오프
        
    Returns:
        지표 이름과 값이 포함된 Dict
    """
    map_scores = []
    recall_scores = []
    precision_scores = []
    ndcg_scores = []
    
    for user_id in ground_truth:
        true_items = set(ground_truth[user_id])
        
        if user_id not in predictions:
            # 이 사용자에 대한 예측이 없음
            map_scores.append(0.0)
            recall_scores.append(0.0)
            precision_scores.append(0.0)
            ndcg_scores.append(0.0)
            continue
        
        pred_items = predictions[user_id][:k]
        
        # MAP@K
        ap = 0.0
        num_hits = 0.0
        for i, item in enumerate(pred_items):
            if item in true_items:
                num_hits += 1.0
                ap += num_hits / (i + 1.0)
        map_scores.append(ap / min(len(true_items), k))
        
        # Recall@K
        hits = sum(1 for item in pred_items if item in true_items)
        recall_scores.append(hits / len(true_items) if true_items else 0.0)
        
        # Precision@K
        precision_scores.append(hits / len(pred_items) if pred_items else 0.0)
        
        # NDCG@K
        dcg = 0.0
        for i, item in enumerate(pred_items):
            if item in true_items:
                dcg += 1.0 / np.log2(i + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    
    return {
        f'map@{k}': np.mean(map_scores),
        f'recall@{k}': np.mean(recall_scores),
        f'precision@{k}': np.mean(precision_scores),
        f'ndcg@{k}': np.mean(ndcg_scores)
    }