import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

class HnMDataset(Dataset):
    def __init__(self, df, user_map, item_map, all_item_ids, num_neg_samples=5):
        self.df = df
        self.user_map = user_map
        self.item_map = item_map
        self.all_item_ids = all_item_ids
        self.num_neg_samples = num_neg_samples

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id_str = row['customer_id']
        item_id_str = row['article_id']

        user_idx = self.user_map.get(user_id_str)
        item_idx = self.item_map.get(item_id_str)

        # For validation/test, we might encounter users/items not in the training map
        if user_idx is None or item_idx is None:
            # Handle unseen items/users, e.g., by returning a special index or skipping
            # For simplicity, we'll return -1 and handle it in the model/evaluation
            user_idx = user_idx if user_idx is not None else -1
            item_idx = item_idx if item_idx is not None else -1
            neg_item_indices = [-1] * self.num_neg_samples
        else:
            # Negative sampling for training
            neg_item_indices = []
            while len(neg_item_indices) < self.num_neg_samples:
                neg_item_id = np.random.choice(self.all_item_ids)
                if neg_item_id != item_id_str: # Ensure it's not the positive item
                    neg_item_idx = self.item_map.get(neg_item_id)
                    if neg_item_idx is not None:
                        neg_item_indices.append(neg_item_idx)

        return {
            'user_id': user_idx,
            'pos_item_id': item_idx,
            'neg_item_id': np.array(neg_item_indices, dtype=np.int64)
        }

class HnMLightningDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=1024, val_days=7, test_days=7, num_neg_samples=5, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_days = val_days
        self.test_days = test_days
        self.num_neg_samples = num_neg_samples
        self.num_workers = num_workers

    def setup(self, stage=None):
        # --- 1. Load Data ---
        transactions_df = pd.read_csv(
            f'{self.data_dir}/transactions_train.csv',
            dtype={'article_id': str},
            parse_dates=['t_dat']
        )

        # --- 2. Time-based Split ---
        end_date = transactions_df['t_dat'].max()
        test_start_date = end_date - pd.Timedelta(days=self.test_days)
        val_start_date = test_start_date - pd.Timedelta(days=self.val_days)

        self.train_df = transactions_df[transactions_df['t_dat'] < val_start_date]
        self.val_df = transactions_df[(transactions_df['t_dat'] >= val_start_date) & (transactions_df['t_dat'] < test_start_date)]
        self.test_df = transactions_df[transactions_df['t_dat'] >= test_start_date]

        print(f"Train set: {self.train_df['t_dat'].min()} to {self.train_df['t_dat'].max()} ({len(self.train_df)} rows)")
        print(f"Validation set: {self.val_df['t_dat'].min()} to {self.val_df['t_dat'].max()} ({len(self.val_df)} rows)")
        print(f"Test set: {self.test_df['t_dat'].min()} to {self.test_df['t_dat'].max()} ({len(self.test_df)} rows)")

        # --- 3. Create Mappings based on Training Data Only ---
        train_users = self.train_df['customer_id'].unique()
        train_items = self.train_df['article_id'].unique()
        
        self.user_map = {user_id: i for i, user_id in enumerate(train_users)}
        self.item_map = {item_id: i for i, item_id in enumerate(train_items)}
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        self.all_training_item_ids = list(self.item_map.keys())
        self.item_map_inv = {v: k for k, v in self.item_map.items()}

        # Create edge_index for LightGCN
        # Users are 0 to num_users-1, Items are num_users to num_users+num_items-1
        train_interactions_mapped = self.train_df.copy()
        train_interactions_mapped['user_idx'] = train_interactions_mapped['customer_id'].map(self.user_map)
        train_interactions_mapped['item_idx'] = train_interactions_mapped['article_id'].map(self.item_map)

        # Filter out any interactions where user/item might not be in the map (should be rare for train_df)
        train_interactions_mapped.dropna(subset=['user_idx', 'item_idx'], inplace=True)

        edge_index_users = torch.tensor(train_interactions_mapped['user_idx'].values, dtype=torch.long)
        edge_index_items = torch.tensor(train_interactions_mapped['item_idx'].values + self.num_users, dtype=torch.long)
        self.edge_index = torch.stack([edge_index_users, edge_index_items], dim=0)

        # Make it bidirectional for LightGCN
        self.edge_index = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1)

    def train_dataloader(self):
        dataset = HnMDataset(
            self.train_df, self.user_map, self.item_map, 
            self.all_training_item_ids, self.num_neg_samples
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = HnMDataset(self.val_df, self.user_map, self.item_map, [], 0) # No negative sampling for val
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        dataset = HnMDataset(self.test_df, self.user_map, self.item_map, [], 0) # No negative sampling for test
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
