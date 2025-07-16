"""추천을 위한 LightGCN 모델."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List
import torch_sparse
from torch_sparse import SparseTensor
import numpy as np
from ..evaluation import RecommendationMetrics


class LightGCN(pl.LightningModule):
    """추천을 위한 경량 그래프 컨볼루션 네트워크.
    
    특징 변환과 비선형 활성화를 제거하여 GCN을 단순화합니다.
    그래프 컨볼루션에서 임베딩의 정규화된 합만을 사용합니다.
    
    참고문헌: He et al. "LightGCN: Simplifying and Powering Graph Convolution 
    Network for Recommendation" (SIGIR 2020)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        top_k: int = 12,
        alpha: Optional[float] = None,
    ):
        """LightGCN 모델 초기화.
        
        Args:
            num_users: 고유 사용자 수
            num_items: 고유 아이템 수
            embedding_dim: 사용자/아이템 임베딩 차원
            num_layers: GCN 레이어 수
            learning_rate: 옵틴마이저의 학습률
            weight_decay: L2 정규화 가중치
            top_k: 추천할 아이템 수
            alpha: 레이어 결합을 위한 선택적 가중치 요소. None인 경우 균등 가중치 사용
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.top_k = top_k
        
        # 레이어 결합 가중치
        if alpha is None:
            # 균등 가중치
            self.alpha = [1.0 / (num_layers + 1)] * (num_layers + 1)
        else:
            # 지수 감쇠
            self.alpha = [alpha ** i for i in range(num_layers + 1)]
            # 정규화
            alpha_sum = sum(self.alpha)
            self.alpha = [a / alpha_sum for a in self.alpha]
        
        # 사용자와 아이템에 대한 임베딩 (하나의 행렬로 결합)
        self.embeddings = nn.Embedding(self.num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        
        # 그래프 인접 행렬 (데이터 모듈에 의해 설정됨)
        self.graph = None
        self.edge_index = None
        self.edge_weight = None
        
        # 메트릭
        self.metrics = RecommendationMetrics(top_k=top_k)
    
    def set_graph(self, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        """그래프 구조 설정.
        
        Args:
            edge_index: 엣지 인덱스 [2, num_edges]
            edge_weight: 선택적 엣지 가중치 [num_edges]
        """
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
        # 정규화된 인접 행렬 생성
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # 자기 루프 추가
        num_nodes = self.num_nodes
        edge_index, edge_weight = self._add_self_loops(
            edge_index, edge_weight, num_nodes
        )
        
        # 대칭 정규화
        row, col = edge_index
        deg = torch_sparse.sum(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        # 정규화된 그래프 저장
        self.graph = SparseTensor(
            row=row, col=col, value=edge_weight,
            sparse_sizes=(num_nodes, num_nodes)
        )
    
    def _add_self_loops(
        self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """그래프에 자기 루프 추가.
        
        Args:
            edge_index: 엣지 인덱스
            edge_weight: 엣지 가중치
            num_nodes: 노드 수
            
        Returns:
            업데이트된 edge_index와 edge_weight
        """
        loop_index = torch.arange(num_nodes, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(num_nodes, device=edge_weight.device)
        
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_weight = torch.cat([edge_weight, loop_weight])
        
        return edge_index, edge_weight
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """GCN 레이어를 통한 순전파.
        
        Returns:
            최종 사용자 임베딩 [num_users, embedding_dim]
            최종 아이템 임베딩 [num_items, embedding_dim]
        """
        if self.graph is None:
            raise RuntimeError("Graph not set. Call set_graph() first.")
        
        # 모든 임베딩 가져오기
        all_embeddings = self.embeddings.weight
        embeddings_list = [all_embeddings]
        
        # 레이어를 통해 전파
        for layer in range(self.num_layers):
            all_embeddings = self.graph @ all_embeddings
            embeddings_list.append(all_embeddings)
        
        # 모든 레이어의 임베딩 결합
        final_embeddings = torch.zeros_like(embeddings_list[0])
        for i, embeddings in enumerate(embeddings_list):
            final_embeddings += self.alpha[i] * embeddings
        
        # 사용자와 아이템 임베딩으로 분할
        user_embeddings = final_embeddings[:self.num_users]
        item_embeddings = final_embeddings[self.num_users:]
        
        return user_embeddings, item_embeddings
    
    def predict(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """사용자-아이템 쌍에 대한 점수 예측.
        
        Args:
            user_ids: 사용자 ID 텐서 [batch_size]
            item_ids: 아이템 ID 텐서 [batch_size]
            
        Returns:
            예측 점수 [batch_size]
        """
        user_embeddings, item_embeddings = self.forward()
        
        user_embeds = user_embeddings[user_ids]
        item_embeds = item_embeddings[item_ids]
        
        # 내적
        scores = (user_embeds * item_embeds).sum(dim=1)
        
        return scores
    
    def predict_all_items(self, user_ids: torch.Tensor) -> torch.Tensor:
        """주어진 사용자에 대해 모든 아이템의 점수 예측.
        
        Args:
            user_ids: 사용자 ID 텐서 [batch_size]
            
        Returns:
            모든 아이템에 대한 점수 [batch_size, num_items]
        """
        user_embeddings, item_embeddings = self.forward()
        
        user_embeds = user_embeddings[user_ids]  # [batch_size, embedding_dim]
        
        # 모든 아이템과의 점수 계산
        scores = torch.matmul(user_embeds, item_embeddings.t())  # [batch_size, num_items]
        
        return scores
    
    def bpr_loss(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor, neg_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """BPR (베이지안 개인화 랭킹) 손실 계산.
        
        Args:
            user_ids: 사용자 ID [batch_size]
            pos_item_ids: 양성 아이템 ID [batch_size]
            neg_item_ids: 음성 아이템 ID [batch_size]
            
        Returns:
            BPR 손실 값
        """
        # 정규화를 위해 전파 전 임베딩 가져오기
        user_embeds_0 = self.embeddings(user_ids)
        pos_item_embeds_0 = self.embeddings(pos_item_ids + self.num_users)
        neg_item_embeds_0 = self.embeddings(neg_item_ids + self.num_users)
        
        # 전파된 임베딩 가져오기
        user_embeddings, item_embeddings = self.forward()
        
        user_embeds = user_embeddings[user_ids]
        pos_item_embeds = item_embeddings[pos_item_ids]
        neg_item_embeds = item_embeddings[neg_item_ids]
        
        # 점수 계산
        pos_scores = (user_embeds * pos_item_embeds).sum(dim=1)
        neg_scores = (user_embeds * neg_item_embeds).sum(dim=1)
        
        # BPR 손실
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        
        # 임베딩에 대한 L2 정규화 (전파 전)
        reg_loss = self.weight_decay * (
            user_embeds_0.norm(2).pow(2) +
            pos_item_embeds_0.norm(2).pow(2) +
            neg_item_embeds_0.norm(2).pow(2)
        ) / user_embeds_0.size(0)
        
        return loss + reg_loss
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """학습 단계.
        
        Args:
            batch: user_ids, pos_items, neg_items를 포함하는 배치
            batch_idx: 배치 인덱스
            
        Returns:
            손실 값
        """
        user_ids = batch['user_ids']
        pos_items = batch['pos_items']
        neg_items = batch['neg_items']
        
        # BPR 손실 계산
        loss = self.bpr_loss(user_ids, pos_items, neg_items)
        
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
        """옵틴마이저 구성."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0  # 손실에서 정규화를 처리함
        )
        
        # 학습률 스케줄러
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_map_at_k',
                'frequency': 1
            }
        }
    
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