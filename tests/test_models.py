"""Test cases for recommendation models."""
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
    """Common model configuration."""
    return {
        'num_users': 100,
        'num_items': 50,
        'embedding_dim': 16,
        'top_k': 5,
    }


class TestMatrixFactorization:
    """Test Matrix Factorization model."""
    
    def test_model_initialization(self, model_config):
        """Test model initialization."""
        model = MatrixFactorization(**model_config)
        assert model.num_users == 100
        assert model.num_items == 50
        assert model.embedding_dim == 16
    
    def test_forward_pass(self, model_config):
        """Test forward pass."""
        model = MatrixFactorization(**model_config)
        
        batch_size = 10
        user_ids = torch.randint(0, 100, (batch_size,))
        item_ids = torch.randint(0, 50, (batch_size,))
        
        predictions = model(user_ids, item_ids)
        assert predictions.shape == (batch_size,)
    
    def test_predict_all_items(self, model_config):
        """Test prediction for all items."""
        model = MatrixFactorization(**model_config)
        
        batch_size = 5
        user_ids = torch.randint(0, 100, (batch_size,))
        
        scores = model.predict_all_items(user_ids)
        assert scores.shape == (batch_size, 50)
    
    def test_recommend(self, model_config):
        """Test recommendation generation."""
        model = MatrixFactorization(**model_config)
        
        user_ids = torch.tensor([0, 1, 2])
        recommendations = model.recommend(user_ids)
        
        assert recommendations.shape == (3, 5)
        assert recommendations.max() < 50
        assert recommendations.min() >= 0


class TestNeuralCF:
    """Test Neural Collaborative Filtering model."""
    
    def test_model_initialization(self, model_config):
        """Test model initialization."""
        config = model_config.copy()
        config['mlp_dims'] = [32, 16, 8]
        model = NeuralCF(**config)
        
        assert model.num_users == 100
        assert model.num_items == 50
        assert len(model.mlp_layers) > 0
    
    def test_forward_pass(self, model_config):
        """Test forward pass."""
        config = model_config.copy()
        config['mlp_dims'] = [32, 16, 8]
        model = NeuralCF(**config)
        
        batch_size = 10
        user_ids = torch.randint(0, 100, (batch_size,))
        item_ids = torch.randint(0, 50, (batch_size,))
        
        predictions = model(user_ids, item_ids)
        assert predictions.shape == (batch_size,)
    
    def test_mlp_architecture(self):
        """Test MLP architecture construction."""
        model = NeuralCF(
            num_users=100,
            num_items=50,
            mf_dim=32,
            mlp_dims=[64, 32, 16],
            top_k=5
        )
        
        # Check MLP layers
        linear_layers = [m for m in model.mlp_layers if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3
        assert linear_layers[0].in_features == 64
        assert linear_layers[-1].out_features == 16


class TestWideDeep:
    """Test Wide & Deep model."""
    
    def test_model_initialization(self, model_config):
        """Test model initialization."""
        config = model_config.copy()
        config['deep_layers'] = [128, 64, 32]
        model = WideDeep(**config)
        
        assert model.num_users == 100
        assert model.num_items == 50
        assert model.use_wide_user_item == True
    
    def test_forward_pass_without_features(self, model_config):
        """Test forward pass without additional features."""
        config = model_config.copy()
        config['deep_layers'] = [64, 32]
        model = WideDeep(**config)
        
        batch_size = 10
        user_ids = torch.randint(0, 100, (batch_size,))
        item_ids = torch.randint(0, 50, (batch_size,))
        
        predictions = model(user_ids, item_ids)
        assert predictions.shape == (batch_size,)
    
    def test_forward_pass_with_features(self):
        """Test forward pass with additional features."""
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
    """Test LightGCN model."""
    
    def test_model_initialization(self, model_config):
        """Test model initialization."""
        config = model_config.copy()
        config['num_layers'] = 3
        model = LightGCN(**config)
        
        assert model.num_users == 100
        assert model.num_items == 50
        assert model.num_layers == 3
        assert len(model.alpha) == 4  # num_layers + 1
    
    def test_graph_setup(self, model_config):
        """Test graph setup."""
        model = LightGCN(**model_config)
        
        # Create a simple bipartite graph
        num_edges = 200
        user_ids = torch.randint(0, 100, (num_edges,))
        item_ids = torch.randint(0, 50, (num_edges,)) + 100  # Offset for items
        
        edge_index = torch.stack([
            torch.cat([user_ids, item_ids]),
            torch.cat([item_ids, user_ids])
        ])
        
        model.set_graph(edge_index)
        assert model.graph is not None
    
    def test_forward_with_graph(self, model_config):
        """Test forward pass with graph."""
        model = LightGCN(**model_config)
        
        # Set up graph
        num_edges = 200
        user_ids = torch.randint(0, 100, (num_edges,))
        item_ids = torch.randint(0, 50, (num_edges,)) + 100
        
        edge_index = torch.stack([
            torch.cat([user_ids, item_ids]),
            torch.cat([item_ids, user_ids])
        ])
        
        model.set_graph(edge_index)
        
        # Forward pass
        user_embeddings, item_embeddings = model.forward()
        assert user_embeddings.shape == (100, 16)
        assert item_embeddings.shape == (50, 16)
    
    def test_bpr_loss(self, model_config):
        """Test BPR loss computation."""
        model = LightGCN(**model_config)
        
        # Set up graph
        num_edges = 200
        user_ids = torch.randint(0, 100, (num_edges,))
        item_ids = torch.randint(0, 50, (num_edges,)) + 100
        
        edge_index = torch.stack([
            torch.cat([user_ids, item_ids]),
            torch.cat([item_ids, user_ids])
        ])
        
        model.set_graph(edge_index)
        
        # Compute BPR loss
        batch_size = 32
        batch_users = torch.randint(0, 100, (batch_size,))
        pos_items = torch.randint(0, 50, (batch_size,))
        neg_items = torch.randint(0, 50, (batch_size,))
        
        loss = model.bpr_loss(batch_users, pos_items, neg_items)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar