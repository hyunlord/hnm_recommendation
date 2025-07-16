"""추천 모델에 대한 테스트 케이스."""
import pytest
import torch
import torch.nn as nn
from src.models import (
    PopularityBaseline,
    MatrixFactorization,
    NeuralCF,
    WideDeep,
    LightGCN
)


@pytest.fixture
def model_config():
    """공통 모델 설정."""
    return {
        'num_users': 100,
        'num_items': 50,
        'embedding_dim': 16,
        'top_k': 5,
    }


class TestMatrixFactorization:
    """행렬 분해 모델 테스트."""
    
    def test_model_initialization(self, model_config):
        """모델 초기화 테스트."""
        model = MatrixFactorization(**model_config)
        assert model.num_users == 100
        assert model.num_items == 50
        assert model.embedding_dim == 16
    
    def test_forward_pass(self, model_config):
        """순전파 테스트."""
        model = MatrixFactorization(**model_config)
        
        batch_size = 10
        user_ids = torch.randint(0, 100, (batch_size,))
        item_ids = torch.randint(0, 50, (batch_size,))
        
        predictions = model(user_ids, item_ids)
        assert predictions.shape == (batch_size,)
    
    def test_predict_all_items(self, model_config):
        """모든 아이템에 대한 예측 테스트."""
        model = MatrixFactorization(**model_config)
        
        batch_size = 5
        user_ids = torch.randint(0, 100, (batch_size,))
        
        scores = model.predict_all_items(user_ids)
        assert scores.shape == (batch_size, 50)
    
    def test_recommend(self, model_config):
        """추천 생성 테스트."""
        model = MatrixFactorization(**model_config)
        
        user_ids = torch.tensor([0, 1, 2])
        recommendations = model.recommend(user_ids)
        
        assert recommendations.shape == (3, 5)
        assert recommendations.max() < 50
        assert recommendations.min() >= 0


class TestNeuralCF:
    """신경망 협업 필터링 모델 테스트."""
    
    def test_model_initialization(self, model_config):
        """모델 초기화 테스트."""
        config = model_config.copy()
        config['mlp_dims'] = [32, 16, 8]
        model = NeuralCF(**config)
        
        assert model.num_users == 100
        assert model.num_items == 50
        assert len(model.mlp_layers) > 0
    
    def test_forward_pass(self, model_config):
        """순전파 테스트."""
        config = model_config.copy()
        config['mlp_dims'] = [32, 16, 8]
        model = NeuralCF(**config)
        
        batch_size = 10
        user_ids = torch.randint(0, 100, (batch_size,))
        item_ids = torch.randint(0, 50, (batch_size,))
        
        predictions = model(user_ids, item_ids)
        assert predictions.shape == (batch_size,)
    
    def test_mlp_architecture(self):
        """MLP 아키텍처 구성 테스트."""
        model = NeuralCF(
            num_users=100,
            num_items=50,
            mf_dim=32,
            mlp_dims=[64, 32, 16],
            top_k=5
        )
        
        # MLP 레이어 확인
        linear_layers = [m for m in model.mlp_layers if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3
        assert linear_layers[0].in_features == 64
        assert linear_layers[-1].out_features == 16


class TestWideDeep:
    """Wide & Deep 모델 테스트."""
    
    def test_model_initialization(self, model_config):
        """모델 초기화 테스트."""
        config = model_config.copy()
        config['deep_layers'] = [128, 64, 32]
        model = WideDeep(**config)
        
        assert model.num_users == 100
        assert model.num_items == 50
        assert model.use_wide_user_item == True
    
    def test_forward_pass_without_features(self, model_config):
        """추가 특성 없이 순전파 테스트."""
        config = model_config.copy()
        config['deep_layers'] = [64, 32]
        model = WideDeep(**config)
        
        batch_size = 10
        user_ids = torch.randint(0, 100, (batch_size,))
        item_ids = torch.randint(0, 50, (batch_size,))
        
        predictions = model(user_ids, item_ids)
        assert predictions.shape == (batch_size,)
    
    def test_forward_pass_with_features(self):
        """추가 특성과 함께 순전파 테스트."""
        model = WideDeep(
            num_users=100,
            num_items=50,
            num_user_features=10,
            num_item_features=20,
            embedding_dim=16,
            deep_layers=[64, 32],
            top_k=5
        )
        
        batch_size = 10
        user_ids = torch.randint(0, 100, (batch_size,))
        item_ids = torch.randint(0, 50, (batch_size,))
        user_features = torch.randn(batch_size, 10)
        item_features = torch.randn(batch_size, 20)
        
        predictions = model(user_ids, item_ids, user_features, item_features)
        assert predictions.shape == (batch_size,)


class TestLightGCN:
    """LightGCN 모델 테스트."""
    
    def test_model_initialization(self, model_config):
        """모델 초기화 테스트."""
        config = model_config.copy()
        config['num_layers'] = 3
        model = LightGCN(**config)
        
        assert model.num_users == 100
        assert model.num_items == 50
        assert model.num_layers == 3
        assert len(model.alpha) == 4  # num_layers + 1
    
    def test_graph_setup(self, model_config):
        """그래프 설정 테스트."""
        model = LightGCN(**model_config)
        
        # 간단한 이분 그래프 생성
        num_edges = 200
        user_ids = torch.randint(0, 100, (num_edges,))
        item_ids = torch.randint(0, 50, (num_edges,)) + 100  # 아이템에 대한 오프셋
        
        edge_index = torch.stack([
            torch.cat([user_ids, item_ids]),
            torch.cat([item_ids, user_ids])
        ])
        
        model.set_graph(edge_index)
        assert model.graph is not None
    
    def test_forward_with_graph(self, model_config):
        """그래프와 함께 순전파 테스트."""
        model = LightGCN(**model_config)
        
        # 그래프 설정
        num_edges = 200
        user_ids = torch.randint(0, 100, (num_edges,))
        item_ids = torch.randint(0, 50, (num_edges,)) + 100
        
        edge_index = torch.stack([
            torch.cat([user_ids, item_ids]),
            torch.cat([item_ids, user_ids])
        ])
        
        model.set_graph(edge_index)
        
        # 순전파
        user_embeddings, item_embeddings = model.forward()
        assert user_embeddings.shape == (100, 16)
        assert item_embeddings.shape == (50, 16)
    
    def test_bpr_loss(self, model_config):
        """BPR 손실 계산 테스트."""
        model = LightGCN(**model_config)
        
        # 그래프 설정
        num_edges = 200
        user_ids = torch.randint(0, 100, (num_edges,))
        item_ids = torch.randint(0, 50, (num_edges,)) + 100
        
        edge_index = torch.stack([
            torch.cat([user_ids, item_ids]),
            torch.cat([item_ids, user_ids])
        ])
        
        model.set_graph(edge_index)
        
        # BPR 손실 계산
        batch_size = 32
        batch_users = torch.randint(0, 100, (batch_size,))
        pos_items = torch.randint(0, 50, (batch_size,))
        neg_items = torch.randint(0, 50, (batch_size,))
        
        loss = model.bpr_loss(batch_users, pos_items, neg_items)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # 스칼라