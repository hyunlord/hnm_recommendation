"""Test baseline model."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytorch_lightning as pl
from src.data import HMDataModule
from src.models import PopularityBaseline
from src.evaluation.metrics import evaluate_recommendations
from src.utils.logger import setup_logger

logger = setup_logger("test_baseline")


def test_baseline():
    """Test popularity baseline model."""
    logger.info("Testing Popularity Baseline Model...")
    
    # Create DataModule
    logger.info("\nCreating DataModule...")
    dm = HMDataModule(
        batch_size=128,
        num_workers=0,
        sample_frac=0.01,  # Use 1% of data for testing
        force_preprocess=False  # Use existing processed data if available
    )
    
    dm.setup()
    
    # Create baseline model
    logger.info("\nCreating baseline model...")
    model = PopularityBaseline(
        n_items=dm.num_items,
        k=12,
        time_decay=0.01,  # Small time decay
        personalized=True  # Filter out already seen items
    )
    
    # Fit popularity on training data
    logger.info("\nFitting popularity on training data...")
    model.fit_popularity(dm.train_df)
    
    # Test forward pass
    logger.info("\nTesting forward pass...")
    import torch
    test_users = torch.tensor([0, 1, 2, 3, 4])
    recommendations = model(test_users)
    logger.info(f"Recommendations shape: {recommendations.shape}")
    logger.info(f"Sample recommendations: {recommendations[0].tolist()}")
    
    # Create trainer for validation
    logger.info("\nRunning validation...")
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='cpu',  # CPU only for baseline
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    # Validate
    val_results = trainer.validate(model, dm.val_dataloader())
    
    logger.info("\nValidation Results:")
    for metric, value in val_results[0].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Test batch recommendation
    logger.info("\nTesting batch recommendations...")
    test_user_ids = dm.val_df['customer_idx'].unique()[:100].tolist()
    recommendations_dict = model.recommend(test_user_ids, k=12)
    
    logger.info(f"Generated recommendations for {len(recommendations_dict)} users")
    
    # Evaluate using our metrics function
    logger.info("\nEvaluating recommendations...")
    
    # Get ground truth
    ground_truth = dm.val_df.groupby('customer_idx')['article_idx'].apply(list).to_dict()
    
    # Filter to test users
    ground_truth_subset = {uid: ground_truth.get(uid, []) for uid in test_user_ids}
    
    # Evaluate
    metrics = evaluate_recommendations(recommendations_dict, ground_truth_subset, k=12)
    
    logger.info("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\n✅ Baseline test completed successfully!")


def test_different_baselines():
    """Test different baseline configurations."""
    logger.info("\n\nTesting Different Baseline Configurations...")
    
    # Load data
    dm = HMDataModule(
        batch_size=128,
        num_workers=0,
        sample_frac=0.01,
        force_preprocess=False
    )
    dm.setup()
    
    configurations = [
        {"name": "Global Popularity", "time_decay": 0.0, "personalized": False},
        {"name": "Time-weighted Popularity", "time_decay": 0.01, "personalized": False},
        {"name": "Personalized Popularity", "time_decay": 0.0, "personalized": True},
        {"name": "Personalized + Time-weighted", "time_decay": 0.01, "personalized": True},
    ]
    
    results = []
    
    for config in configurations:
        logger.info(f"\nTesting {config['name']}...")
        
        model = PopularityBaseline(
            n_items=dm.num_items,
            k=12,
            time_decay=config['time_decay'],
            personalized=config['personalized']
        )
        
        model.fit_popularity(dm.train_df)
        
        # Get recommendations for validation users
        val_users = dm.val_df['customer_idx'].unique()[:1000].tolist()
        recommendations = model.recommend(val_users, k=12)
        
        # Get ground truth
        ground_truth = dm.val_df.groupby('customer_idx')['article_idx'].apply(list).to_dict()
        ground_truth_subset = {uid: ground_truth.get(uid, []) for uid in val_users}
        
        # Evaluate
        metrics = evaluate_recommendations(recommendations, ground_truth_subset, k=12)
        
        result = {"config": config['name'], **metrics}
        results.append(result)
        
        logger.info(f"  MAP@12: {metrics['map@12']:.4f}")
        logger.info(f"  Recall@12: {metrics['recall@12']:.4f}")
    
    # Print comparison
    logger.info("\n\nBaseline Comparison:")
    logger.info("-" * 70)
    logger.info(f"{'Configuration':<30} {'MAP@12':>10} {'Recall@12':>10} {'Precision@12':>10}")
    logger.info("-" * 70)
    
    for result in results:
        logger.info(
            f"{result['config']:<30} "
            f"{result['map@12']:>10.4f} "
            f"{result['recall@12']:>10.4f} "
            f"{result['precision@12']:>10.4f}"
        )
    
    logger.info("-" * 70)
    
    logger.info("\n✅ Baseline comparison completed!")


if __name__ == "__main__":
    test_baseline()
    test_different_baselines()