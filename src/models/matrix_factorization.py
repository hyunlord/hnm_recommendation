"""추천을 위한 행렬 분해 모델."""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..evaluation import RecommendationMetrics


class MatrixFactorization(pl.LightningModule):
    """경사 하강법을 사용하는 행렬 분해 모델.
    
    이 모델은 행렬 분해를 통해 사용자와 아이템의 잠재 요인을 학습합니다.
    예측 등급은 사용자와 아이템 임베딩의 내적으로 계산됩니다.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        top_k: int = 12,
        sparse: bool = True,
    ):
        """행렬 분해 모델 초기화.
        
        Args:
            num_users: 고유 사용자 수
            num_items: 고유 아이템 수
            embedding_dim: 사용자/아이템 임베딩 차원
            learning_rate: 옵티마이저의 학습률
            weight_decay: L2 정규화 가중치
            top_k: 추천할 아이템 수
            sparse: 희소 임베딩 사용 여부
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.top_k = top_k
        
        # 사용자 및 아이템 임베딩
        if sparse:
            self.user_embeddings = nn.Embedding(
                num_users, embedding_dim, sparse=True
            )
            self.item_embeddings = nn.Embedding(
                num_items, embedding_dim, sparse=True
            )
        else:
            self.user_embeddings = nn.Embedding(num_users, embedding_dim)
            self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # 사용자 및 아이템 편향
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # 전역 편향
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 임베딩 초기화
        self._init_weights()
        
        # 메트릭
        self.metrics = RecommendationMetrics(top_k=top_k)
        
    def _init_weights(self):
        """모델 가중치 초기화."""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """등급 예측을 위한 순전파.
        
        Args:
            user_ids: 사용자 ID 텐서 [batch_size]
            item_ids: 아이템 ID 텐서 [batch_size]
            
        Returns:
            예측 등급 [batch_size]
        """
        # 임베딩 가져오기
        user_embeds = self.user_embeddings(user_ids)  # [batch_size, embedding_dim]
        item_embeds = self.item_embeddings(item_ids)  # [batch_size, embedding_dim]
        
        # 편향 가져오기
        user_bias = self.user_bias(user_ids).squeeze()  # [batch_size]
        item_bias = self.item_bias(item_ids).squeeze()  # [batch_size]
        
        # 내적 계산
        dot_product = (user_embeds * item_embeds).sum(dim=1)  # [batch_size]
        
        # 편향 추가
        prediction = dot_product + user_bias + item_bias + self.global_bias
        
        return prediction
    
    def predict_all_items(self, user_ids: torch.Tensor) -> torch.Tensor:
        """주어진 사용자에 대해 모든 아이템의 점수 예측.
        
        Args:
            user_ids: 사용자 ID 텐서 [batch_size]
            
        Returns:
            모든 아이템에 대한 점수 [batch_size, num_items]
        """
        # 사용자 임베딩과 편향 가져오기
        user_embeds = self.user_embeddings(user_ids)  # [batch_size, embedding_dim]
        user_bias = self.user_bias(user_ids)  # [batch_size, 1]
        
        # 모든 아이템 임베딩과 편향 가져오기
        all_item_embeds = self.item_embeddings.weight  # [num_items, embedding_dim]
        all_item_bias = self.item_bias.weight  # [num_items, 1]
        
        # 점수 계산: user_embeds @ all_item_embeds.T
        scores = torch.matmul(user_embeds, all_item_embeds.t())  # [batch_size, num_items]
        
        # 편향 추가
        scores = scores + user_bias + all_item_bias.t() + self.global_bias
        
        return scores
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """학습 단계.
        
        Args:
            batch: user_ids, item_ids, labels를 포함하는 배치
            batch_idx: 배치 인덱스
            
        Returns:
            손실 값
        """
        user_ids = batch['user_ids']
        item_ids = batch['item_ids']
        labels = batch['labels'].float()
        
        # 순전파
        predictions = self(user_ids, item_ids)
        
        # 이진 교차 엔트로피 손실
        loss = nn.functional.binary_cross_entropy_with_logits(
            predictions, labels
        )
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """검증 단계.
        
        Args:
            batch: user_ids와 정답 아이템을 포함하는 배치
            batch_idx: 배치 인덱스
        """
        user_ids = batch['user_ids']
        ground_truth = batch['ground_truth']
        
        # 모든 아이템에 대한 예측 가져오기
        scores = self.predict_all_items(user_ids)
        
        # 상위 k개 아이템 가져오기
        _, top_k_items = torch.topk(scores, self.top_k, dim=1)
        
        # 메트릭 업데이트
        self.metrics.update(top_k_items, ground_truth)
        
    def on_validation_epoch_end(self):
        """검증 메트릭 계산 및 로깅."""
        metrics = self.metrics.compute()
        self.metrics.reset()
        
        for metric_name, value in metrics.items():
            self.log(f'val_{metric_name}', value, prog_bar=True)
            
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """테스트 단계 (검증과 동일)."""
        self.validation_step(batch, batch_idx)
        
    def on_test_epoch_end(self):
        """테스트 메트릭 계산 및 로깅."""
        metrics = self.metrics.compute()
        self.metrics.reset()
        
        for metric_name, value in metrics.items():
            self.log(f'test_{metric_name}', value)
            
    def configure_optimizers(self):
        """옵티마이저 구성."""
        if self.hparams.sparse:
            # 희소 임베딩을 위해 SparseAdam 사용
            optimizer = torch.optim.SparseAdam(
                [
                    {'params': self.user_embeddings.parameters()},
                    {'params': self.item_embeddings.parameters()},
                    {'params': self.user_bias.parameters()},
                    {'params': self.item_bias.parameters()},
                    {'params': [self.global_bias]},
                ],
                lr=self.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        return optimizer
    
    def recommend(
        self, user_ids: torch.Tensor, filter_items: Optional[Dict[int, set]] = None
    ) -> torch.Tensor:
        """사용자를 위한 추천 생성.
        
        Args:
            user_ids: 사용자 ID 텐서
            filter_items: user_id를 필터링할 아이템 집합에 매핑하는 선택적 딕셔너리
            
        Returns:
            각 사용자에 대한 상위 k개 추천 아이템
        """
        self.eval()
        with torch.no_grad():
            scores = self.predict_all_items(user_ids)
            
            # 필요한 경우 아이템 필터링
            if filter_items is not None:
                for i, user_id in enumerate(user_ids.tolist()):
                    if user_id in filter_items:
                        items_to_filter = list(filter_items[user_id])
                        scores[i, items_to_filter] = float('-inf')
            
            # 상위 k개 아이템 가져오기
            _, top_k_items = torch.topk(scores, self.top_k, dim=1)
            
        return top_k_items