"""신경망 협업 필터링 (NCF) 모델."""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List
from ..evaluation import RecommendationMetrics


class NeuralCF(pl.LightningModule):
    """신경망 협업 필터링 모델.
    
    일반화된 행렬 분해(GMF)와 다층 퍼셉트론(MLP)을 결합하여
    비선형 사용자-아이템 상호작용을 학습합니다.
    
    참고문헌: He et al. "Neural Collaborative Filtering" (WWW 2017)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        mf_dim: int = 64,
        mlp_dims: List[int] = [128, 64, 32],
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        top_k: int = 12,
        use_pretrain: bool = False,
    ):
        """NCF 모델 초기화.
        
        Args:
            num_users: 고유 사용자 수
            num_items: 고유 아이템 수
            mf_dim: 행렬 분해 임베딩 차원
            mlp_dims: MLP의 은닉층 차원 리스트
            dropout: 드롭아웃 비율
            learning_rate: 옵티마이저의 학습률
            weight_decay: L2 정규화 가중치
            top_k: 추천할 아이템 수
            use_pretrain: 사전 학습된 GMF/MLP 임베딩 사용 여부
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.top_k = top_k
        
        # GMF 임베딩
        self.gmf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, mf_dim)
        
        # MLP 임베딩
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_dims[0] // 2)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_dims[0] // 2)
        
        # MLP 레이어
        self.mlp_layers = self._build_mlp(mlp_dims, dropout)
        
        # 최종 예측 레이어
        self.prediction_layer = nn.Linear(mf_dim + mlp_dims[-1], 1)
        
        # 가중치 초기화
        self._init_weights()
        
        # 메트릭
        self.metrics = RecommendationMetrics(top_k=top_k)
        
    def _build_mlp(self, dims: List[int], dropout: float) -> nn.Sequential:
        """MLP 레이어 구축.
        
        Args:
            dims: 은닉층 차원 리스트
            dropout: 드롭아웃 비율
            
        Returns:
            순차적 MLP 레이어
        """
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """모델 가중치 초기화."""
        # GMF 임베딩 - smaller initialization
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        
        # MLP 임베딩 - Xavier initialization
        nn.init.xavier_uniform_(self.mlp_user_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_item_embedding.weight)
        
        # MLP 레이어 - Xavier initialization
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # 예측 레이어
        nn.init.xavier_uniform_(self.prediction_layer.weight)
        nn.init.zeros_(self.prediction_layer.bias)
    
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
        # GMF 부분
        gmf_user = self.gmf_user_embedding(user_ids)  # [batch_size, mf_dim]
        gmf_item = self.gmf_item_embedding(item_ids)  # [batch_size, mf_dim]
        gmf_output = gmf_user * gmf_item  # 요소별 곱셈
        
        # MLP 부분
        mlp_user = self.mlp_user_embedding(user_ids)  # [batch_size, mlp_dim/2]
        mlp_item = self.mlp_item_embedding(item_ids)  # [batch_size, mlp_dim/2]
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)  # [batch_size, mlp_dim]
        mlp_output = self.mlp_layers(mlp_input)  # [batch_size, mlp_dims[-1]]
        
        # GMF와 MLP 출력 연결
        concat = torch.cat([gmf_output, mlp_output], dim=1)
        
        # 최종 예측
        prediction = self.prediction_layer(concat).squeeze()
        
        return prediction
    
    def predict_all_items(self, user_ids: torch.Tensor) -> torch.Tensor:
        """주어진 사용자에 대해 모든 아이템의 점수 예측.
        
        Args:
            user_ids: 사용자 ID 텐서 [batch_size]
            
        Returns:
            모든 아이템에 대한 점수 [batch_size, num_items]
        """
        batch_size = user_ids.size(0)
        
        # 사용자 임베딩 가져오기
        gmf_user = self.gmf_user_embedding(user_ids)  # [batch_size, mf_dim]
        mlp_user = self.mlp_user_embedding(user_ids)  # [batch_size, mlp_dim/2]
        
        # 모든 아이템 임베딩 가져오기
        all_items = torch.arange(self.num_items, device=user_ids.device)
        gmf_items = self.gmf_item_embedding(all_items)  # [num_items, mf_dim]
        mlp_items = self.mlp_item_embedding(all_items)  # [num_items, mlp_dim/2]
        
        # 모든 아이템에 대한 점수 계산
        scores = []
        
        # 메모리 문제를 피하기 위해 배치로 처리
        item_batch_size = 1000
        for i in range(0, self.num_items, item_batch_size):
            end_idx = min(i + item_batch_size, self.num_items)
            item_batch = torch.arange(i, end_idx, device=user_ids.device)
            
            # 배치를 위해 사용자 임베딩 확장
            batch_gmf_user = gmf_user.unsqueeze(1).expand(
                batch_size, len(item_batch), -1
            )  # [batch_size, item_batch_size, mf_dim]
            batch_mlp_user = mlp_user.unsqueeze(1).expand(
                batch_size, len(item_batch), -1
            )  # [batch_size, item_batch_size, mlp_dim/2]
            
            # 배치에 대한 아이템 임베딩 가져오기
            batch_gmf_items = gmf_items[i:end_idx].unsqueeze(0).expand(
                batch_size, -1, -1
            )  # [batch_size, item_batch_size, mf_dim]
            batch_mlp_items = mlp_items[i:end_idx].unsqueeze(0).expand(
                batch_size, -1, -1
            )  # [batch_size, item_batch_size, mlp_dim/2]
            
            # GMF 부분
            gmf_output = batch_gmf_user * batch_gmf_items
            
            # MLP 부분
            mlp_input = torch.cat([batch_mlp_user, batch_mlp_items], dim=2)
            # MLP를 위한 형태 변경
            mlp_input_flat = mlp_input.view(-1, mlp_input.size(-1))
            mlp_output_flat = self.mlp_layers(mlp_input_flat)
            mlp_output = mlp_output_flat.view(batch_size, len(item_batch), -1)
            
            # 연결 및 예측
            concat = torch.cat([gmf_output, mlp_output], dim=2)
            concat_flat = concat.view(-1, concat.size(-1))
            batch_scores = self.prediction_layer(concat_flat).view(batch_size, -1)
            
            scores.append(batch_scores)
        
        # 모든 점수 연결
        all_scores = torch.cat(scores, dim=1)  # [batch_size, num_items]
        
        return all_scores
    
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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
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