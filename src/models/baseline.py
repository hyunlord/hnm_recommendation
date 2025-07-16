"""베이스라인 추천 모델."""

import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from src.evaluation.metrics import MeanAveragePrecision, RecallAtK, PrecisionAtK, NDCGAtK
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PopularityBaseline(pl.LightningModule):
    """간단한 인기도 기반 베이스라인 모델."""
    
    def __init__(
        self,
        n_items: int,
        k: int = 12,
        time_decay: float = 0.0,
        personalized: bool = False
    ):
        """인기도 베이스라인 초기화.
        
        Args:
            n_items: 아이템 수
            k: 추천 수
            time_decay: 시간 감쇠 계수 (0 = 감쇠 없음)
            personalized: 사용자 이력을 기반으로 개인화할지 여부
        """
        super().__init__()
        self.n_items = n_items
        self.k = k
        self.time_decay = time_decay
        self.personalized = personalized
        
        # 학습 데이터에서 계산됨
        self.item_popularity = None
        self.user_history = {} if personalized else None
        
        # 메트릭
        self.val_map = MeanAveragePrecision(k=k)
        self.val_recall = RecallAtK(k=k)
        self.val_precision = PrecisionAtK(k=k)
        self.val_ndcg = NDCGAtK(k=k)
        
        # 하이퍼파라미터 저장
        self.save_hyperparameters()
    
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """사용자에 대한 추천 가져오기.
        
        Args:
            user_ids: 사용자 인덱스 [batch_size]
            
        Returns:
            추천 아이템 [batch_size, k]
        """
        batch_size = user_ids.shape[0]
        recommendations = torch.zeros(batch_size, self.k, dtype=torch.long)
        
        if self.item_popularity is None:
            # 학습되지 않은 경우 랜덤 아이템 반환
            for i in range(batch_size):
                recommendations[i] = torch.randperm(self.n_items)[:self.k]
        else:
            # 상위 k개 인기 아이템 가져오기
            top_items = torch.tensor(self.item_popularity[:self.k])
            
            if not self.personalized:
                # 모든 사용자에게 동일한 추천
                recommendations = top_items.unsqueeze(0).repeat(batch_size, 1)
            else:
                # 사용자가 이미 상호작용한 아이템 필터링
                for i in range(batch_size):
                    user_id = user_ids[i].item()
                    if user_id in self.user_history:
                        # 사용자 이력에 없는 아이템 가져오기
                        user_items = set(self.user_history[user_id])
                        available_items = [item for item in self.item_popularity 
                                         if item not in user_items]
                        recommendations[i] = torch.tensor(available_items[:self.k])
                    else:
                        recommendations[i] = top_items
        
        return recommendations
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """학습 단계 - 인기도 계산."""
        # 인기도 베이스라인의 경우 실제로 학습하지 않음
        # 더미 손실 반환
        return torch.tensor(0.0, requires_grad=True)
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """검증 단계."""
        users = batch['user']
        true_items = batch['items']
        items_mask = batch['items_mask']
        
        # 추천 가져오기
        pred_items = self(users)
        
        # 메트릭 업데이트
        self.val_map.update(pred_items, true_items, items_mask)
        self.val_recall.update(pred_items, true_items, items_mask)
        self.val_precision.update(pred_items, true_items, items_mask)
        self.val_ndcg.update(pred_items, true_items, items_mask)
        
        return {'val_loss': torch.tensor(0.0)}
    
    def on_validation_epoch_end(self):
        """검증 메트릭 로깅."""
        map_score = self.val_map.compute()
        recall_score = self.val_recall.compute()
        precision_score = self.val_precision.compute()
        ndcg_score = self.val_ndcg.compute()
        
        self.log('val/map@12', map_score, prog_bar=True)
        self.log('val/recall@12', recall_score)
        self.log('val/precision@12', precision_score)
        self.log('val/ndcg@12', ndcg_score)
        
        # 메트릭 초기화
        self.val_map.reset()
        self.val_recall.reset()
        self.val_precision.reset()
        self.val_ndcg.reset()
    
    def configure_optimizers(self):
        """베이스라인에는 옵티마이저가 필요 없음."""
        # Lightning 요구사항을 충족하기 위한 더미 옵티마이저 반환
        return torch.optim.SGD(self.parameters(), lr=0.0)
    
    def fit_popularity(self, train_df: pd.DataFrame, date_col: str = 't_dat'):
        """학습 데이터에서 인기도 계산.
        
        Args:
            train_df: [customer_idx, article_idx, t_dat] 칼럼을 가진 학습 데이터프레임
            date_col: 날짜 칼럼 이름
        """
        logger.info("Computing item popularity...")
        
        if self.time_decay > 0 and date_col in train_df.columns:
            # 시간 감쇠 적용
            max_date = train_df[date_col].max()
            train_df['days_ago'] = (max_date - train_df[date_col]).dt.days
            train_df['weight'] = np.exp(-self.time_decay * train_df['days_ago'])
            
            # 가중치가 적용된 인기도
            popularity = train_df.groupby('article_idx')['weight'].sum()
        else:
            # 단순 카운트
            popularity = train_df['article_idx'].value_counts()
        
        # 인기도 순으로 정렬
        self.item_popularity = popularity.sort_values(ascending=False).index.tolist()
        
        # 개인화된 경우 사용자 이력 저장
        if self.personalized:
            self.user_history = train_df.groupby('customer_idx')['article_idx'].apply(list).to_dict()
        
        logger.info(f"Computed popularity for {len(self.item_popularity)} items")
    
    def recommend(self, user_ids: List[int], k: Optional[int] = None) -> Dict[int, List[int]]:
        """여러 사용자에 대한 추천 가져오기.
        
        Args:
            user_ids: 사용자 ID 리스트
            k: 추천 수 (기본값: self.k)
            
        Returns:
            user_id를 추천 아이템 리스트에 매핑하는 딕셔너리
        """
        if k is None:
            k = self.k
        
        recommendations = {}
        
        for user_id in user_ids:
            if self.personalized and user_id in self.user_history:
                # 이미 본 아이템 필터링
                user_items = set(self.user_history[user_id])
                recs = [item for item in self.item_popularity 
                       if item not in user_items][:k]
            else:
                # 상위 인기 아이템 반환
                recs = self.item_popularity[:k]
            
            recommendations[user_id] = recs
        
        return recommendations