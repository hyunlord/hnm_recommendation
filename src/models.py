import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn.models import LightGCN
from torch_geometric.data import Data

class NeuMF(pl.LightningModule):
    def __init__(self, num_users, num_items, mf_dim=32, mlp_layer_sizes=[64, 32, 16], lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # --- GMF (Generalized Matrix Factorization) Layers ---
        self.mf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.mf_item_embedding = nn.Embedding(num_items, mf_dim)

        # --- MLP (Multi-Layer Perceptron) Layers ---
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_layer_sizes[0] // 2)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_layer_sizes[0] // 2)

        mlp_layers = []
        for i in range(len(mlp_layer_sizes) - 1):
            mlp_layers.append(nn.Linear(mlp_layer_sizes[i], mlp_layer_sizes[i+1]))
            mlp_layers.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_layers)

        # --- Final Prediction Layer ---
        # The input size is the concatenation of GMF and MLP outputs
        predict_size = mf_dim + mlp_layer_sizes[-1]
        self.predict_layer = nn.Linear(predict_size, 1)

    def forward(self, user_id, item_id):
        # GMF part
        mf_user_embed = self.mf_user_embedding(user_id)
        mf_item_embed = self.mf_item_embedding(item_id)
        gmf_output = mf_user_embed * mf_item_embed

        # MLP part
        mlp_user_embed = self.mlp_user_embedding(user_id)
        mlp_item_embed = self.mlp_item_embedding(item_id)
        mlp_input = torch.cat([mlp_user_embed, mlp_item_embed], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        # Concatenate GMF and MLP outputs
        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)

        # Final prediction
        prediction = self.predict_layer(concat_output)
        return prediction.squeeze()

    def training_step(self, batch, batch_idx):
        user_id = batch['user_id']
        pos_item_id = batch['pos_item_id']
        neg_item_id = batch['neg_item_id'] # Shape: (batch_size, num_neg_samples)

        # Positive item scores
        pos_scores = self.forward(user_id, pos_item_id)

        # Negative item scores
        # We need to repeat user_id to match the shape of neg_item_id
        user_id_repeated = user_id.unsqueeze(1).repeat(1, neg_item_id.shape[1])
        neg_scores = self.forward(user_id_repeated.view(-1), neg_item_id.view(-1))
        neg_scores = neg_scores.view(neg_item_id.shape[0], -1)

        # BPR Loss
        # We want pos_scores > neg_scores. The loss is -log(sigmoid(pos_scores - neg_scores))
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores)).mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # For validation, we can simply calculate the loss on the validation set
        # More complex metrics like MAP@12 would be calculated in on_validation_epoch_end
        user_id = batch['user_id']
        pos_item_id = batch['pos_item_id']
        
        # In validation, we don't use negative samples from the dataloader
        # We just need to check the model's ability to score positive items high
        # A simple validation loss can be calculated if we had labels, but for implicit we just log loss
        # Let's calculate a dummy loss for now, the real evaluation will be done later
        
        # Filter out any users/items that were not in the training set (marked as -1)
        valid_indices = (user_id != -1) & (pos_item_id != -1)
        if not valid_indices.any():
            return
            
        user_id = user_id[valid_indices]
        pos_item_id = pos_item_id[valid_indices]

        pos_scores = self.forward(user_id, pos_item_id)
        # A simple loss could be to maximize the positive score
        loss = -pos_scores.mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class LightGCN_PL(pl.LightningModule):
    def __init__(self, num_nodes, num_users, num_items, embedding_dim=64, num_layers=3, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = LightGCN(num_nodes, embedding_dim, num_layers=num_layers)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

    def forward(self, edge_index):
        return self.model(edge_index)

    def training_step(self, batch, batch_idx):
        user_indices = batch['user_id']
        pos_item_indices = batch['pos_item_id']
        neg_item_indices = batch['neg_item_id']

        # Get embeddings
        # Note: LightGCN's get_embedding method requires the full graph (edge_index)
        # which is stored in the datamodule.
        user_embed, item_embed = self.model.get_embedding(self.trainer.datamodule.edge_index.to(self.device))
        
        # BPR Loss calculation
        # Ensure item_embed indices are shifted to match LightGCN's internal item indexing
        # (which starts after user indices)
        pos_scores = (user_embed[user_indices] * item_embed[pos_item_indices + self.num_users]).sum(dim=1)
        neg_scores = (user_embed[user_indices].unsqueeze(1) * item_embed[neg_item_indices + self.num_users]).sum(dim=2).view(-1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_indices = batch['user_id']
        pos_item_indices = batch['pos_item_id']

        valid_indices = (user_indices != -1) & (pos_item_indices != -1)
        if not valid_indices.any():
            return
            
        user_indices = user_indices[valid_indices]
        pos_item_indices = pos_item_indices[valid_indices]

        user_embed, item_embed = self.model.get_embedding(self.trainer.datamodule.edge_index.to(self.device))
        scores = (user_embed[user_indices] * item_embed[pos_item_indices + self.num_users]).sum(dim=1)
        loss = -scores.mean() # Dummy loss for validation
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
