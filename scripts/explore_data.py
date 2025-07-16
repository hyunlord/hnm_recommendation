"""Quick data exploration script to understand the H&M dataset."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.constants import *
from src.utils.logger import setup_logger

logger = setup_logger("data_exploration")


def explore_articles():
    """Explore articles (products) data."""
    logger.info("\n=== ARTICLES DATA EXPLORATION ===")
    
    # Load data
    articles_df = pd.read_csv(ARTICLES_PATH)
    logger.info(f"Articles shape: {articles_df.shape}")
    
    # Basic info
    logger.info(f"\nColumns: {list(articles_df.columns)}")
    logger.info(f"\nMissing values:\n{articles_df.isnull().sum()[articles_df.isnull().sum() > 0]}")
    
    # Key statistics
    logger.info(f"\nUnique products: {articles_df['article_id'].nunique():,}")
    logger.info(f"Product types: {articles_df['product_type_name'].nunique()}")
    logger.info(f"Product groups: {articles_df['product_group_name'].nunique()}")
    logger.info(f"Departments: {articles_df['department_name'].nunique()}")
    
    # Top categories
    logger.info("\nTop 10 Product Groups:")
    for idx, (group, count) in enumerate(articles_df['product_group_name'].value_counts().head(10).items()):
        logger.info(f"  {idx+1}. {group}: {count:,} products")
    
    logger.info("\nTop 10 Product Types:")
    for idx, (ptype, count) in enumerate(articles_df['product_type_name'].value_counts().head(10).items()):
        logger.info(f"  {idx+1}. {ptype}: {count:,} products")
    
    logger.info("\nDepartment Distribution:")
    for dept, count in articles_df['department_name'].value_counts().items():
        logger.info(f"  {dept}: {count:,} products ({count/len(articles_df)*100:.1f}%)")
    
    logger.info("\nColor Distribution (Top 10):")
    for idx, (color, count) in enumerate(articles_df['colour_group_name'].value_counts().head(10).items()):
        logger.info(f"  {idx+1}. {color}: {count:,} products")
    
    # Sample product
    logger.info("\nSample product:")
    sample = articles_df.iloc[0]
    for col, val in sample.items():
        if pd.notna(val):
            logger.info(f"  {col}: {val}")
    
    return articles_df


def explore_customers():
    """Explore customers data."""
    logger.info("\n\n=== CUSTOMERS DATA EXPLORATION ===")
    
    # Load data
    customers_df = pd.read_csv(CUSTOMERS_PATH)
    logger.info(f"Customers shape: {customers_df.shape}")
    
    # Basic info
    logger.info(f"\nColumns: {list(customers_df.columns)}")
    logger.info(f"\nMissing values:\n{customers_df.isnull().sum()[customers_df.isnull().sum() > 0]}")
    
    # Age analysis
    age_stats = customers_df['age'].describe()
    logger.info(f"\nAge statistics:")
    for stat, value in age_stats.items():
        logger.info(f"  {stat}: {value:.1f}")
    
    # Missing age
    missing_age = customers_df['age'].isnull().sum()
    logger.info(f"\nMissing age values: {missing_age:,} ({missing_age/len(customers_df)*100:.1f}%)")
    
    # Customer segments
    logger.info(f"\nClub member status:")
    for status, count in customers_df['club_member_status'].value_counts().items():
        logger.info(f"  {status}: {count:,} ({count/len(customers_df)*100:.1f}%)")
    
    logger.info(f"\nFashion news frequency:")
    for freq, count in customers_df['fashion_news_frequency'].value_counts().items():
        logger.info(f"  {freq}: {count:,} ({count/len(customers_df)*100:.1f}%)")
    
    logger.info(f"\nActive status:")
    for status, count in customers_df['Active'].value_counts().items():
        logger.info(f"  {status}: {count:,} ({count/len(customers_df)*100:.1f}%)")
    
    # Age groups
    customers_with_age = customers_df.dropna(subset=['age'])
    age_groups = pd.cut(customers_with_age['age'], 
                        bins=[0, 20, 30, 40, 50, 60, 100], 
                        labels=['<20', '20-29', '30-39', '40-49', '50-59', '60+'])
    logger.info(f"\nAge group distribution:")
    for group, count in age_groups.value_counts().sort_index().items():
        logger.info(f"  {group}: {count:,} ({count/len(customers_with_age)*100:.1f}%)")
    
    return customers_df


def explore_transactions():
    """Explore transactions data (using sampling due to size)."""
    logger.info("\n\n=== TRANSACTIONS DATA EXPLORATION ===")
    
    # First, get basic file info
    total_lines = sum(1 for _ in open(TRANSACTIONS_PATH)) - 1
    logger.info(f"Total transactions: {total_lines:,}")
    
    # Load a sample
    sample_size = 1_000_000
    logger.info(f"\nLoading {sample_size:,} sample transactions...")
    trans_df = pd.read_csv(TRANSACTIONS_PATH, nrows=sample_size)
    trans_df['t_dat'] = pd.to_datetime(trans_df['t_dat'])
    
    logger.info(f"Sample shape: {trans_df.shape}")
    logger.info(f"Columns: {list(trans_df.columns)}")
    
    # Date range
    logger.info(f"\nDate range: {trans_df['t_dat'].min()} to {trans_df['t_dat'].max()}")
    date_span = (trans_df['t_dat'].max() - trans_df['t_dat'].min()).days
    logger.info(f"Date span: {date_span} days ({date_span/365:.1f} years)")
    
    # Unique counts
    logger.info(f"\nUnique customers in sample: {trans_df['customer_id'].nunique():,}")
    logger.info(f"Unique articles in sample: {trans_df['article_id'].nunique():,}")
    
    # Price statistics
    price_stats = trans_df['price'].describe()
    logger.info(f"\nPrice statistics:")
    for stat, value in price_stats.items():
        logger.info(f"  {stat}: ${value:.2f}")
    
    # Sales channels
    logger.info(f"\nSales channel distribution:")
    for channel, count in trans_df['sales_channel_id'].value_counts().items():
        logger.info(f"  Channel {channel}: {count:,} ({count/len(trans_df)*100:.1f}%)")
    
    # Customer purchase frequency (in sample)
    customer_counts = trans_df['customer_id'].value_counts()
    logger.info(f"\nCustomer purchase frequency (sample):")
    logger.info(f"  Mean: {customer_counts.mean():.2f}")
    logger.info(f"  Median: {customer_counts.median():.0f}")
    logger.info(f"  Max: {customer_counts.max()}")
    logger.info(f"  Customers with 1 purchase: {(customer_counts == 1).sum():,} ({(customer_counts == 1).sum()/len(customer_counts)*100:.1f}%)")
    
    # Product popularity (in sample)
    product_counts = trans_df['article_id'].value_counts()
    logger.info(f"\nProduct popularity (sample):")
    logger.info(f"  Mean purchases per product: {product_counts.mean():.2f}")
    logger.info(f"  Median purchases per product: {product_counts.median():.0f}")
    logger.info(f"  Max purchases for a product: {product_counts.max()}")
    
    logger.info(f"\nTop 10 most purchased products (sample):")
    for idx, (product, count) in enumerate(product_counts.head(10).items()):
        logger.info(f"  {idx+1}. Article {product}: {count} purchases")
    
    # Recent activity
    last_month = trans_df['t_dat'].max() - timedelta(days=30)
    recent_trans = trans_df[trans_df['t_dat'] >= last_month]
    logger.info(f"\nLast 30 days activity (in sample):")
    logger.info(f"  Transactions: {len(recent_trans):,}")
    logger.info(f"  Active customers: {recent_trans['customer_id'].nunique():,}")
    logger.info(f"  Active products: {recent_trans['article_id'].nunique():,}")
    
    return trans_df


def calculate_sparsity(n_users, n_items, n_interactions):
    """Calculate sparsity of the interaction matrix."""
    total_possible = n_users * n_items
    sparsity = 1 - (n_interactions / total_possible)
    return sparsity * 100


def main():
    """Run all explorations."""
    logger.info("Starting H&M data exploration...")
    
    # Explore each dataset
    articles_df = explore_articles()
    customers_df = explore_customers()
    trans_df = explore_transactions()
    
    # Overall statistics
    logger.info("\n\n=== OVERALL DATASET STATISTICS ===")
    
    # Calculate total statistics
    total_customers = len(customers_df)
    total_articles = len(articles_df)
    total_transactions = sum(1 for _ in open(TRANSACTIONS_PATH)) - 1
    
    logger.info(f"\nDataset size:")
    logger.info(f"  Total customers: {total_customers:,}")
    logger.info(f"  Total articles: {total_articles:,}")
    logger.info(f"  Total transactions: {total_transactions:,}")
    
    # Sparsity
    sparsity = calculate_sparsity(total_customers, total_articles, total_transactions)
    logger.info(f"\nInteraction matrix sparsity: {sparsity:.4f}%")
    logger.info(f"Average transactions per customer: {total_transactions/total_customers:.2f}")
    logger.info(f"Average transactions per article: {total_transactions/total_articles:.2f}")
    
    # Recommendations implications
    logger.info("\n\n=== KEY INSIGHTS FOR RECOMMENDATION SYSTEM ===")
    logger.info("1. Extremely sparse data (>99.99%) - collaborative filtering will be challenging")
    logger.info("2. Many customers have very few purchases - cold start problem")
    logger.info("3. Strong product popularity bias - popular items baseline will be important")
    logger.info("4. Rich product metadata available - content-based filtering promising")
    logger.info("5. Customer demographics available - can help with cold start")
    logger.info("6. Time-based patterns exist - sequential models may be effective")
    logger.info("7. Multiple sales channels - consider channel-specific behavior")
    
    logger.info("\nData exploration completed!")


if __name__ == "__main__":
    main()