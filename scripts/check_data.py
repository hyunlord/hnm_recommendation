"""Script to check if all required data files are present and show basic statistics."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.utils.constants import (
    DATA_DIR, ARTICLES_PATH, CUSTOMERS_PATH, 
    TRANSACTIONS_PATH, SUBMISSION_PATH, IMAGES_DIR
)
from src.utils.logger import setup_logger


def check_data_files():
    """Check if all required data files exist and show basic information."""
    logger = setup_logger("data_check")
    
    logger.info("Checking data files...")
    logger.info(f"Data directory: {DATA_DIR}")
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        return False
    
    # Check each required file
    required_files = {
        "Articles": ARTICLES_PATH,
        "Customers": CUSTOMERS_PATH,
        "Transactions": TRANSACTIONS_PATH,
        "Sample Submission": SUBMISSION_PATH
    }
    
    all_files_exist = True
    for name, path in required_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {name}: {path.name} ({size_mb:.1f} MB)")
            
            # Show basic stats for CSV files
            if path.suffix == '.csv' and size_mb < 500:  # Only load smaller files
                try:
                    df = pd.read_csv(path, nrows=5)
                    logger.info(f"  Columns: {', '.join(df.columns)}")
                    logger.info(f"  Shape preview: {df.shape}")
                except Exception as e:
                    logger.warning(f"  Could not read file: {e}")
        else:
            logger.error(f"✗ {name}: {path.name} NOT FOUND")
            all_files_exist = False
    
    # Check images directory
    if IMAGES_DIR.exists():
        image_count = sum(1 for _ in IMAGES_DIR.rglob("*.jpg"))
        logger.info(f"✓ Images directory: {image_count} images found")
    else:
        logger.warning(f"✗ Images directory not found: {IMAGES_DIR}")
    
    # Show transaction file info (handle large file)
    if TRANSACTIONS_PATH.exists():
        logger.info("\nTransaction file statistics (sampling first 10000 rows):")
        try:
            df_sample = pd.read_csv(TRANSACTIONS_PATH, nrows=10000)
            logger.info(f"  Columns: {', '.join(df_sample.columns)}")
            logger.info(f"  Date range (sample): {df_sample['t_dat'].min()} to {df_sample['t_dat'].max()}")
            logger.info(f"  Unique customers (sample): {df_sample['customer_id'].nunique()}")
            logger.info(f"  Unique articles (sample): {df_sample['article_id'].nunique()}")
        except Exception as e:
            logger.error(f"  Could not read transaction file: {e}")
    
    return all_files_exist


if __name__ == "__main__":
    success = check_data_files()
    if success:
        print("\n✅ All required data files are present!")
    else:
        print("\n❌ Some data files are missing. Please download the H&M dataset from Kaggle.")
        print("Visit: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data")