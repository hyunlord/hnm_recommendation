"""Test the DataModule implementation."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.data import HMDataModule
from src.utils.logger import setup_logger

logger = setup_logger("test_datamodule")


def test_datamodule():
    """Test the DataModule with a small sample."""
    logger.info("Testing H&M DataModule...")
    
    # Create DataModule with small sample for testing
    dm = HMDataModule(
        batch_size=32,
        num_workers=0,  # Set to 0 for testing
        sample_frac=0.001,  # Use 0.1% of data for quick test
        negative_samples=2,
        use_features=True,
        force_preprocess=True  # Force preprocessing for test
    )
    
    # Prepare data
    logger.info("Preparing data...")
    dm.prepare_data()
    
    # Setup
    logger.info("Setting up datasets...")
    dm.setup()
    
    # Print statistics
    logger.info(f"\nDataModule Statistics:")
    logger.info(f"  Number of users: {dm.num_users:,}")
    logger.info(f"  Number of items: {dm.num_items:,}")
    logger.info(f"  Number of user features: {dm.num_user_features}")
    logger.info(f"  Number of item features: {dm.num_item_features}")
    
    # Test train dataloader
    logger.info("\nTesting train dataloader...")
    train_loader = dm.train_dataloader()
    
    for i, batch in enumerate(train_loader):
        if i == 0:
            logger.info(f"  Batch keys: {list(batch.keys())}")
            logger.info(f"  User shape: {batch['user'].shape}")
            logger.info(f"  Item shape: {batch['item'].shape}")
            logger.info(f"  Label shape: {batch['label'].shape}")
            
            if 'user_features' in batch:
                logger.info(f"  User features shape: {batch['user_features'].shape}")
            if 'item_features' in batch:
                logger.info(f"  Item features shape: {batch['item_features'].shape}")
            
            # Check data types
            assert batch['user'].dtype == torch.long
            assert batch['item'].dtype == torch.long
            assert batch['label'].dtype == torch.float
            
            # Check label distribution
            pos_ratio = batch['label'].mean().item()
            logger.info(f"  Positive ratio in batch: {pos_ratio:.2f}")
        
        if i >= 2:  # Check first 3 batches
            break
    
    logger.info(f"  Total train batches: {len(train_loader)}")
    
    # Test validation dataloader
    logger.info("\nTesting validation dataloader...")
    val_loader = dm.val_dataloader()
    
    for i, batch in enumerate(val_loader):
        if i == 0:
            logger.info(f"  Batch keys: {list(batch.keys())}")
            logger.info(f"  User shape: {batch['user'].shape}")
            logger.info(f"  Items shape: {batch['items'].shape}")
            logger.info(f"  Items mask shape: {batch['items_mask'].shape}")
            
            # Check that mask works correctly
            n_valid_items = batch['items_mask'].sum(dim=1)
            logger.info(f"  Valid items per user: min={n_valid_items.min()}, max={n_valid_items.max()}")
        
        if i >= 2:
            break
    
    logger.info(f"  Total validation batches: {len(val_loader)}")
    
    # Test popular items
    logger.info("\nTesting popular items retrieval...")
    popular_week = dm.get_popular_items(k=12, period='week')
    popular_month = dm.get_popular_items(k=12, period='month')
    
    logger.info(f"  Popular items (week): {popular_week}")
    logger.info(f"  Popular items (month): {popular_month}")
    
    logger.info("\n✅ DataModule test completed successfully!")


def test_sequential_datamodule():
    """Test sequential dataset variant."""
    logger.info("\n\nTesting Sequential DataModule...")
    
    # Create sequential DataModule
    dm = HMDataModule(
        batch_size=16,
        num_workers=0,
        sample_frac=0.001,
        sequential=True,
        max_seq_length=20,
        force_preprocess=False  # Use existing processed data
    )
    
    dm.setup()
    
    # Test sequential train dataloader
    logger.info("\nTesting sequential train dataloader...")
    train_loader = dm.train_dataloader()
    
    for i, batch in enumerate(train_loader):
        if i == 0:
            logger.info(f"  Batch keys: {list(batch.keys())}")
            logger.info(f"  User shape: {batch['user'].shape}")
            logger.info(f"  Input sequence shape: {batch['input_seq'].shape}")
            logger.info(f"  Target shape: {batch['target'].shape}")
            logger.info(f"  Sequence length shape: {batch['seq_len'].shape}")
            
            # Check sequence properties
            logger.info(f"  Max sequence length: {batch['input_seq'].shape[1]}")
            logger.info(f"  Actual sequence lengths: {batch['seq_len'][:5].tolist()}")
        
        if i >= 2:
            break
    
    logger.info(f"  Total sequential train batches: {len(train_loader)}")
    
    logger.info("\n✅ Sequential DataModule test completed successfully!")


if __name__ == "__main__":
    test_datamodule()
    test_sequential_datamodule()