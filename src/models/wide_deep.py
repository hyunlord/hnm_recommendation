"""추천을 위한 Wide & Deep 모델."""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
from ..evaluation import RecommendationMetrics


class WideDeep(pl.LightningModule):
    """추천을 위한 Wide & Deep 학습 모델.
    
    암기(wide 컴포넌트)와 일반화(deep 컴포넌트)를 결합합니다.
    Wide 컴포넌트는 교차 곱 변환을 통해 특징 상호작용을 포착합니다.
    Deep 컴포넌트는 신경망을 통해 고차원 특징 상호작용을 학습합니다.
    
    참고문헌: Cheng et al. "Wide & Deep Learning for Recommender Systems" (RecSys 2016)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_user_features: int = 0,
        num_item_features: int = 0,
        embedding_dim: int = 64,
        deep_layers: List[int] = [512, 256, 128],
        dropout: float = 0.1,
        use_wide_user_item: bool = True,
        use_wide_features: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        top_k: int = 12,
    ):
        """Wide & Deep 모델 초기화.
        
        Args:
            num_users: 고유 사용자 수
            num_items: 고유 아이템 수
            num_user_features: 추가 사용자 특징 수
            num_item_features: 추가 아이템 특징 수
            embedding_dim: 임베딩 차원
            deep_layers: deep 컴포넌트의 은닉층 차원 리스트
            dropout: 드롭아웃 비율
            use_wide_user_item: wide 컴포넌트에서 사용자-아이템 상호작용 사용 여부
            use_wide_features: wide 컴포넌트에서 특징 교차 사용 여부
            learning_rate: 옵티마이저의 학습률
            weight_decay: L2 정규화 가중치
            top_k: 추천할 아이템 수
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_user_features = num_user_features
        self.num_item_features = num_item_features
        self.embedding_dim = embedding_dim
        self.deep_layers = deep_layers
        self.dropout = dropout
        self.use_wide_user_item = use_wide_user_item
        self.use_wide_features = use_wide_features
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.top_k = top_k
        
        # Wide component
        self._build_wide_component()
        
        # Deep component
        self._build_deep_component()
        
        # Final prediction layer
        wide_dim = self._calculate_wide_dim()
        deep_dim = deep_layers[-1]
        self.final_layer = nn.Linear(wide_dim + deep_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
        # Metrics
        self.metrics = RecommendationMetrics(top_k=top_k)
    
    def _calculate_wide_dim(self) -> int:
        """Calculate dimension of wide component output."""
        dim = 0
        if self.use_wide_user_item:
            dim += self.num_users + self.num_items
        if self.use_wide_features:
            dim += self.num_user_features + self.num_item_features
        return dim
    
    def _build_wide_component(self):
        """Build wide component."""
        # wide 컴포넌트를 위한 사용자 및 아이템 임베딩 (원-핫 인코딩)
        if self.use_wide_user_item:
            self.wide_user_embedding = nn.Embedding(self.num_users, 1)
            self.wide_item_embedding = nn.Embedding(self.num_items, 1)
        
        # wide 컴포넌트를 위한 특징 임베딩
        if self.use_wide_features and self.num_user_features > 0:
            self.wide_user_features = nn.Linear(self.num_user_features, self.num_user_features)
        if self.use_wide_features and self.num_item_features > 0:
            self.wide_item_features = nn.Linear(self.num_item_features, self.num_item_features)
    
    def _build_deep_component(self):
        """Build deep component."""
        # User and item embeddings for deep component
        self.deep_user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.deep_item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        
        # Feature embeddings for deep component
        if self.num_user_features > 0:
            self.deep_user_features = nn.Linear(self.num_user_features, self.embedding_dim)
        if self.num_item_features > 0:
            self.deep_item_features = nn.Linear(self.num_item_features, self.embedding_dim)
        
        # Calculate input dimension for deep network
        deep_input_dim = 2 * self.embedding_dim  # user + item embeddings
        if self.num_user_features > 0:
            deep_input_dim += self.embedding_dim
        if self.num_item_features > 0:
            deep_input_dim += self.embedding_dim
        
        # Deep network layers
        layers = []
        prev_dim = deep_input_dim
        for hidden_dim in self.deep_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        self.deep_network = nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights."""
        # Wide component
        if self.use_wide_user_item:
            nn.init.xavier_uniform_(self.wide_user_embedding.weight)
            nn.init.xavier_uniform_(self.wide_item_embedding.weight)
        
        # Deep component embeddings
        nn.init.xavier_uniform_(self.deep_user_embedding.weight)
        nn.init.xavier_uniform_(self.deep_item_embedding.weight)
        
        # Deep network
        for layer in self.deep_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Final layer
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for rating prediction.
        
        Args:
            user_ids: Tensor of user IDs [batch_size]
            item_ids: Tensor of item IDs [batch_size]
            user_features: Optional user features [batch_size, num_user_features]
            item_features: Optional item features [batch_size, num_item_features]
            
        Returns:
            Predicted ratings [batch_size]
        """
        batch_size = user_ids.size(0)
        
        # Wide component
        wide_outputs = []
        
        if self.use_wide_user_item:
            # One-hot user and item representations
            wide_user = torch.zeros(batch_size, self.num_users, device=user_ids.device)
            wide_user.scatter_(1, user_ids.unsqueeze(1), 1)
            wide_outputs.append(wide_user)
            
            wide_item = torch.zeros(batch_size, self.num_items, device=item_ids.device)
            wide_item.scatter_(1, item_ids.unsqueeze(1), 1)
            wide_outputs.append(wide_item)
        
        if self.use_wide_features:
            if user_features is not None and self.num_user_features > 0:
                wide_user_feat = self.wide_user_features(user_features)
                wide_outputs.append(wide_user_feat)
            if item_features is not None and self.num_item_features > 0:
                wide_item_feat = self.wide_item_features(item_features)
                wide_outputs.append(wide_item_feat)
        
        if wide_outputs:
            wide_output = torch.cat(wide_outputs, dim=1)
        else:
            wide_output = torch.zeros(batch_size, 0, device=user_ids.device)
        
        # Deep component
        deep_inputs = []
        
        # User and item embeddings
        deep_user = self.deep_user_embedding(user_ids)
        deep_inputs.append(deep_user)
        
        deep_item = self.deep_item_embedding(item_ids)
        deep_inputs.append(deep_item)
        
        # Feature embeddings
        if user_features is not None and self.num_user_features > 0:
            deep_user_feat = self.deep_user_features(user_features)
            deep_inputs.append(deep_user_feat)
        if item_features is not None and self.num_item_features > 0:
            deep_item_feat = self.deep_item_features(item_features)
            deep_inputs.append(deep_item_feat)
        
        deep_input = torch.cat(deep_inputs, dim=1)
        deep_output = self.deep_network(deep_input)
        
        # Combine wide and deep
        combined = torch.cat([wide_output, deep_output], dim=1)
        
        # Final prediction
        prediction = self.final_layer(combined).squeeze()
        
        return prediction
    
    def predict_all_items(
        self, user_ids: torch.Tensor, user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict scores for all items for given users.
        
        Args:
            user_ids: Tensor of user IDs [batch_size]
            user_features: Optional user features [batch_size, num_user_features]
            
        Returns:
            Scores for all items [batch_size, num_items]
        """
        batch_size = user_ids.size(0)
        scores = []
        
        # Process items in batches to avoid memory issues
        item_batch_size = 500
        for i in range(0, self.num_items, item_batch_size):
            end_idx = min(i + item_batch_size, self.num_items)
            batch_items = torch.arange(i, end_idx, device=user_ids.device)
            
            # Repeat user_ids for each item in batch
            expanded_users = user_ids.unsqueeze(1).expand(
                batch_size, len(batch_items)
            ).contiguous().view(-1)
            
            # Repeat items for each user
            expanded_items = batch_items.unsqueeze(0).expand(
                batch_size, -1
            ).contiguous().view(-1)
            
            # Expand user features if provided
            expanded_user_features = None
            if user_features is not None:
                expanded_user_features = user_features.unsqueeze(1).expand(
                    batch_size, len(batch_items), -1
                ).contiguous().view(-1, user_features.size(-1))
            
            # Get predictions
            batch_scores = self(
                expanded_users,
                expanded_items,
                expanded_user_features,
                None  # Item features would need to be loaded separately
            )
            
            # Reshape back to [batch_size, item_batch_size]
            batch_scores = batch_scores.view(batch_size, -1)
            scores.append(batch_scores)
        
        # Concatenate all scores
        all_scores = torch.cat(scores, dim=1)
        
        return all_scores
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch containing user_ids, item_ids, labels, and optional features
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        user_ids = batch['user_ids']
        item_ids = batch['item_ids']
        labels = batch['labels'].float()
        user_features = batch.get('user_features', None)
        item_features = batch.get('item_features', None)
        
        # Forward pass
        predictions = self(user_ids, item_ids, user_features, item_features)
        
        # Binary cross entropy loss
        loss = nn.functional.binary_cross_entropy_with_logits(
            predictions, labels
        )
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step.
        
        Args:
            batch: Batch containing user_ids, ground truth, and optional features
            batch_idx: Batch index
        """
        user_ids = batch['user_ids']
        ground_truth = batch['ground_truth']
        user_features = batch.get('user_features', None)
        
        # Get predictions for all items
        scores = self.predict_all_items(user_ids, user_features)
        
        # Get top-k items
        _, top_k_items = torch.topk(scores, self.top_k, dim=1)
        
        # Update metrics
        self.metrics.update(top_k_items, ground_truth)
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        metrics = self.metrics.compute()
        self.metrics.reset()
        
        for metric_name, value in metrics.items():
            self.log(f'val_{metric_name}', value, prog_bar=True)
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step (same as validation)."""
        self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        metrics = self.metrics.compute()
        self.metrics.reset()
        
        for metric_name, value in metrics.items():
            self.log(f'test_{metric_name}', value)
    
    def configure_optimizers(self):
        """Configure optimizer."""
        # Separate optimizers for wide and deep components
        wide_params = []
        deep_params = []
        
        # Wide component parameters
        if self.use_wide_user_item:
            wide_params.extend(self.wide_user_embedding.parameters())
            wide_params.extend(self.wide_item_embedding.parameters())
        if hasattr(self, 'wide_user_features'):
            wide_params.extend(self.wide_user_features.parameters())
        if hasattr(self, 'wide_item_features'):
            wide_params.extend(self.wide_item_features.parameters())
        
        # Deep component parameters
        deep_params.extend(self.deep_user_embedding.parameters())
        deep_params.extend(self.deep_item_embedding.parameters())
        if hasattr(self, 'deep_user_features'):
            deep_params.extend(self.deep_user_features.parameters())
        if hasattr(self, 'deep_item_features'):
            deep_params.extend(self.deep_item_features.parameters())
        deep_params.extend(self.deep_network.parameters())
        
        # Final layer
        deep_params.extend(self.final_layer.parameters())
        
        # Different learning rates for wide and deep
        optimizer = torch.optim.Adam([
            {'params': wide_params, 'lr': self.learning_rate * 0.1},  # Lower LR for wide
            {'params': deep_params, 'lr': self.learning_rate}
        ], weight_decay=self.weight_decay)
        
        # Learning rate scheduler
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
        self,
        user_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        filter_items: Optional[Dict[int, set]] = None
    ) -> torch.Tensor:
        """Generate recommendations for users.
        
        Args:
            user_ids: Tensor of user IDs
            user_features: Optional user features
            filter_items: Optional dict mapping user_id to set of items to filter out
            
        Returns:
            Top-k recommended items for each user
        """
        self.eval()
        with torch.no_grad():
            scores = self.predict_all_items(user_ids, user_features)
            
            # Filter out items if needed
            if filter_items is not None:
                for i, user_id in enumerate(user_ids.tolist()):
                    if user_id in filter_items:
                        items_to_filter = list(filter_items[user_id])
                        scores[i, items_to_filter] = float('-inf')
            
            # Get top-k items
            _, top_k_items = torch.topk(scores, self.top_k, dim=1)
            
        return top_k_items