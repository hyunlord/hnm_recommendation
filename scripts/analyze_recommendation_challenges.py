"""Analyze specific challenges for recommendation system design."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.utils.constants import *
from src.utils.logger import setup_logger

logger = setup_logger("recommendation_challenges")


def analyze_cold_start_problem():
    """Analyze cold start challenges in detail."""
    logger.info("\n=== COLD START PROBLEM ANALYSIS ===")
    
    # Load transaction sample
    logger.info("Loading data...")
    sample_size = 3_000_000
    trans_df = pd.read_csv(TRANSACTIONS_PATH, nrows=sample_size, parse_dates=['t_dat'])
    customers_df = pd.read_csv(CUSTOMERS_PATH)
    articles_df = pd.read_csv(ARTICLES_PATH)
    
    # Analyze customer cold start
    customer_purchases = trans_df['customer_id'].value_counts()
    
    # Define cold start thresholds
    cold_thresholds = [1, 2, 3, 5, 10]
    logger.info("\nCustomer cold start analysis:")
    for threshold in cold_thresholds:
        cold_customers = (customer_purchases <= threshold).sum()
        cold_pct = cold_customers / len(customer_purchases) * 100
        logger.info(f"  Customers with ≤{threshold} purchases: {cold_customers:,} ({cold_pct:.1f}%)")
    
    # Analyze new customer rate
    trans_df['month'] = trans_df['t_dat'].dt.to_period('M')
    monthly_new_customers = trans_df.groupby('month')['customer_id'].apply(
        lambda x: x[~x.isin(trans_df[trans_df['month'] < x.name]['customer_id'].unique())].nunique()
    )
    
    logger.info("\nNew customers per month:")
    for month, count in monthly_new_customers.items():
        logger.info(f"  {month}: {count:,} new customers")
    
    # Analyze product cold start
    product_purchases = trans_df['article_id'].value_counts()
    
    logger.info("\nProduct cold start analysis:")
    for threshold in cold_thresholds:
        cold_products = (product_purchases <= threshold).sum()
        cold_pct = cold_products / len(product_purchases) * 100
        logger.info(f"  Products with ≤{threshold} purchases: {cold_products:,} ({cold_pct:.1f}%)")
    
    # New product introduction rate
    product_first_sale = trans_df.groupby('article_id')['t_dat'].min()
    new_products_by_month = product_first_sale.dt.to_period('M').value_counts().sort_index()
    
    logger.info("\nNew products introduced per month:")
    for month, count in new_products_by_month.items():
        logger.info(f"  {month}: {count:,} new products")
    
    # Customer demographics for cold start
    logger.info("\nCold start customer demographics:")
    cold_customer_ids = customer_purchases[customer_purchases <= 3].index
    cold_customer_info = customers_df[customers_df['customer_id'].isin(cold_customer_ids)]
    
    # Age distribution for cold start customers
    if 'age' in cold_customer_info.columns:
        cold_age_dist = cold_customer_info['age'].describe()
        logger.info(f"  Cold start customer age: mean={cold_age_dist['mean']:.1f}, median={cold_age_dist['50%']:.0f}")
    
    # Member status for cold start
    cold_member_dist = cold_customer_info['club_member_status'].value_counts()
    for status, count in cold_member_dist.items():
        logger.info(f"  {status}: {count:,} ({count/len(cold_customer_info)*100:.1f}%)")
    
    return customer_purchases, product_purchases


def analyze_popularity_bias():
    """Analyze popularity bias and its implications."""
    logger.info("\n\n=== POPULARITY BIAS ANALYSIS ===")
    
    # Load data
    sample_size = 3_000_000
    trans_df = pd.read_csv(TRANSACTIONS_PATH, nrows=sample_size, parse_dates=['t_dat'])
    
    # Product popularity distribution
    product_popularity = trans_df['article_id'].value_counts()
    
    # Gini coefficient calculation
    def gini_coefficient(x):
        """Calculate Gini coefficient for inequality measurement."""
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((n + 1 - np.arange(1, n + 1)) * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n
    
    gini = gini_coefficient(product_popularity.values)
    logger.info(f"\nProduct popularity Gini coefficient: {gini:.3f}")
    logger.info("(0 = perfect equality, 1 = perfect inequality)")
    
    # Top N% analysis
    total_purchases = product_popularity.sum()
    cumulative_purchases = product_popularity.cumsum()
    
    percentiles = [1, 5, 10, 20, 50]
    logger.info("\nPurchase concentration:")
    for pct in percentiles:
        n_products = int(len(product_popularity) * pct / 100)
        purchases_pct = cumulative_purchases.iloc[n_products-1] / total_purchases * 100
        logger.info(f"  Top {pct}% products account for {purchases_pct:.1f}% of purchases")
    
    # Head vs Tail analysis
    head_size = int(len(product_popularity) * 0.2)  # Top 20%
    head_products = product_popularity.head(head_size)
    tail_products = product_popularity.iloc[head_size:]
    
    logger.info(f"\nHead vs Tail products (80/20 split):")
    logger.info(f"  Head products ({head_size:,}): {head_products.sum():,} purchases ({head_products.sum()/total_purchases*100:.1f}%)")
    logger.info(f"  Tail products ({len(tail_products):,}): {tail_products.sum():,} purchases ({tail_products.sum()/total_purchases*100:.1f}%)")
    
    # Category-wise popularity
    articles_df = pd.read_csv(ARTICLES_PATH)
    trans_with_meta = trans_df.merge(
        articles_df[['article_id', 'product_group_name', 'department_name']], 
        on='article_id'
    )
    
    # Popularity concentration by product group
    logger.info("\nPopularity concentration by product group:")
    for group in trans_with_meta['product_group_name'].value_counts().head(5).index:
        group_products = trans_with_meta[trans_with_meta['product_group_name'] == group]['article_id'].value_counts()
        group_gini = gini_coefficient(group_products.values)
        logger.info(f"  {group}: Gini={group_gini:.3f}, Products={len(group_products):,}")
    
    # Visualization
    save_dir = project_root / 'experiments' / 'visualizations'
    
    plt.figure(figsize=(12, 6))
    
    # Lorenz curve
    plt.subplot(1, 2, 1)
    sorted_purchases = np.sort(product_popularity.values)
    cumsum_purchases = np.cumsum(sorted_purchases) / np.sum(sorted_purchases)
    cumsum_products = np.arange(1, len(sorted_purchases) + 1) / len(sorted_purchases)
    
    plt.plot(cumsum_products, cumsum_purchases, 'b-', label=f'Actual (Gini={gini:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect equality')
    plt.xlabel('Cumulative % of Products')
    plt.ylabel('Cumulative % of Purchases')
    plt.title('Product Popularity Lorenz Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log-log plot
    plt.subplot(1, 2, 2)
    ranks = np.arange(1, len(product_popularity) + 1)
    plt.loglog(ranks, product_popularity.values, 'b.', alpha=0.5)
    plt.xlabel('Product Rank (log scale)')
    plt.ylabel('Purchase Count (log scale)')
    plt.title('Product Popularity Power Law Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'popularity_bias_analysis.png', dpi=150)
    plt.close()
    
    return product_popularity, gini


def analyze_temporal_dynamics():
    """Analyze temporal patterns for recommendation."""
    logger.info("\n\n=== TEMPORAL DYNAMICS ANALYSIS ===")
    
    # Load full date range sample
    logger.info("Loading temporal data...")
    sample_size = 5_000_000
    trans_df = pd.read_csv(TRANSACTIONS_PATH, nrows=sample_size, parse_dates=['t_dat'])
    
    # Product lifecycle patterns
    product_lifecycle = trans_df.groupby('article_id').agg({
        't_dat': ['min', 'max', 'count'],
        'customer_id': 'nunique'
    })
    product_lifecycle.columns = ['first_sale', 'last_sale', 'total_sales', 'unique_customers']
    product_lifecycle['lifespan_days'] = (product_lifecycle['last_sale'] - product_lifecycle['first_sale']).dt.days
    product_lifecycle['sales_per_day'] = product_lifecycle['total_sales'] / (product_lifecycle['lifespan_days'] + 1)
    
    # Categorize products by lifecycle stage
    max_date = trans_df['t_dat'].max()
    product_lifecycle['days_since_last_sale'] = (max_date - product_lifecycle['last_sale']).dt.days
    
    def categorize_lifecycle(row):
        if row['lifespan_days'] < 7:
            return 'Flash'
        elif row['days_since_last_sale'] > 30:
            return 'Declining'
        elif row['lifespan_days'] < 30:
            return 'New'
        else:
            return 'Mature'
    
    product_lifecycle['stage'] = product_lifecycle.apply(categorize_lifecycle, axis=1)
    
    stage_dist = product_lifecycle['stage'].value_counts()
    logger.info("\nProduct lifecycle stages:")
    for stage, count in stage_dist.items():
        logger.info(f"  {stage}: {count:,} products ({count/len(product_lifecycle)*100:.1f}%)")
    
    # Sales velocity patterns
    logger.info("\nSales velocity by lifecycle stage:")
    for stage in ['Flash', 'New', 'Mature', 'Declining']:
        if stage in product_lifecycle['stage'].values:
            stage_products = product_lifecycle[product_lifecycle['stage'] == stage]
            logger.info(f"  {stage}: avg {stage_products['sales_per_day'].mean():.1f} sales/day")
    
    # Weekly patterns
    trans_df['week'] = trans_df['t_dat'].dt.isocalendar().week
    trans_df['year_week'] = trans_df['t_dat'].dt.strftime('%Y-W%U')
    
    # Product introduction patterns
    weekly_new_products = trans_df.groupby('year_week')['article_id'].apply(
        lambda x: x[~x.isin(trans_df[trans_df['year_week'] < x.name]['article_id'].unique())].nunique()
    )
    
    logger.info(f"\nWeekly new product introduction:")
    logger.info(f"  Average new products per week: {weekly_new_products.mean():.1f}")
    logger.info(f"  Max new products in a week: {weekly_new_products.max()}")
    
    # Customer shopping patterns over time
    customer_frequency = trans_df.groupby('customer_id')['t_dat'].agg(['min', 'max', 'count'])
    customer_frequency['days_between'] = (customer_frequency['max'] - customer_frequency['min']).dt.days
    customer_frequency['purchase_frequency'] = customer_frequency['days_between'] / (customer_frequency['count'] - 1)
    customer_frequency = customer_frequency[customer_frequency['count'] > 1]
    
    logger.info(f"\nCustomer shopping frequency:")
    logger.info(f"  Average days between purchases: {customer_frequency['purchase_frequency'].mean():.1f}")
    logger.info(f"  Median days between purchases: {customer_frequency['purchase_frequency'].median():.1f}")
    
    # Trend detection
    weekly_sales = trans_df.groupby('year_week').size()
    weekly_unique_products = trans_df.groupby('year_week')['article_id'].nunique()
    weekly_unique_customers = trans_df.groupby('year_week')['customer_id'].nunique()
    
    # Calculate week-over-week growth
    wow_growth = weekly_sales.pct_change().mean() * 100
    logger.info(f"\nWeekly growth patterns:")
    logger.info(f"  Average week-over-week sales growth: {wow_growth:.1f}%")
    
    return product_lifecycle, customer_frequency


def analyze_customer_item_features():
    """Analyze features useful for content-based filtering."""
    logger.info("\n\n=== CONTENT FEATURES ANALYSIS ===")
    
    # Load metadata
    articles_df = pd.read_csv(ARTICLES_PATH)
    customers_df = pd.read_csv(CUSTOMERS_PATH)
    
    # Article features cardinality
    logger.info("\nArticle feature cardinality:")
    feature_cols = ['product_type_name', 'product_group_name', 'graphical_appearance_name',
                   'colour_group_name', 'department_name', 'section_name', 'garment_group_name']
    
    for col in feature_cols:
        n_unique = articles_df[col].nunique()
        logger.info(f"  {col}: {n_unique} unique values")
    
    # Text description analysis
    desc_available = articles_df['detail_desc'].notna().sum()
    logger.info(f"\nProduct descriptions available: {desc_available:,} ({desc_available/len(articles_df)*100:.1f}%)")
    
    if desc_available > 0:
        # Analyze description length
        desc_lengths = articles_df['detail_desc'].dropna().str.len()
        logger.info(f"  Average description length: {desc_lengths.mean():.1f} characters")
        logger.info(f"  Median description length: {desc_lengths.median():.0f} characters")
    
    # Customer features analysis
    logger.info("\nCustomer feature availability:")
    for col in customers_df.columns:
        if col != 'customer_id':
            available = customers_df[col].notna().sum()
            logger.info(f"  {col}: {available:,} ({available/len(customers_df)*100:.1f}% available)")
    
    # Cross-feature correlations (for articles)
    # Check if certain colors are more common in certain departments
    color_dept_cross = pd.crosstab(articles_df['colour_group_name'], 
                                   articles_df['department_name'])
    
    # Find dominant color for each department
    logger.info("\nDominant colors by department (top 5 departments):")
    top_depts = articles_df['department_name'].value_counts().head(5).index
    for dept in top_depts:
        if dept in color_dept_cross.columns:
            top_color = color_dept_cross[dept].idxmax()
            color_pct = color_dept_cross[dept].max() / color_dept_cross[dept].sum() * 100
            logger.info(f"  {dept}: {top_color} ({color_pct:.1f}%)")
    
    # Feature co-occurrence patterns
    feature_patterns = articles_df.groupby(['product_group_name', 'garment_group_name']).size()
    feature_patterns = feature_patterns.sort_values(ascending=False).head(10)
    
    logger.info("\nCommon feature combinations:")
    for (pg, gg), count in feature_patterns.items():
        logger.info(f"  {pg} + {gg}: {count:,} products")
    
    return articles_df, customers_df


def analyze_recommendation_scenarios():
    """Analyze different recommendation scenarios."""
    logger.info("\n\n=== RECOMMENDATION SCENARIOS ANALYSIS ===")
    
    # Load sample data
    sample_size = 2_000_000
    trans_df = pd.read_csv(TRANSACTIONS_PATH, nrows=sample_size, parse_dates=['t_dat'])
    
    # Scenario 1: New user with demographics
    logger.info("\nScenario 1: New user recommendations")
    logger.info("  Strategy: Use demographic-based popularity")
    logger.info("  Fallback: Global popularity baseline")
    
    # Scenario 2: Existing user with few purchases
    sparse_users = trans_df['customer_id'].value_counts()
    sparse_users = sparse_users[sparse_users <= 5]
    logger.info(f"\nScenario 2: Sparse users ({len(sparse_users):,} users)")
    logger.info("  Strategy: Content-based on purchased items + popularity")
    
    # Scenario 3: Active user
    active_users = trans_df['customer_id'].value_counts()
    active_users = active_users[active_users > 20]
    logger.info(f"\nScenario 3: Active users ({len(active_users):,} users)")
    logger.info("  Strategy: Collaborative filtering + temporal patterns")
    
    # Scenario 4: New product
    product_intro_dates = trans_df.groupby('article_id')['t_dat'].min()
    recent_date = trans_df['t_dat'].max() - timedelta(days=7)
    new_products = product_intro_dates[product_intro_dates > recent_date]
    logger.info(f"\nScenario 4: New products ({len(new_products):,} products)")
    logger.info("  Strategy: Content similarity to popular items in same category")
    
    # Scenario 5: Seasonal recommendations
    logger.info("\nScenario 5: Seasonal recommendations")
    logger.info("  Strategy: Time-aware models with seasonal features")
    
    # Analyze basket completion opportunities
    same_day_purchases = trans_df.groupby(['customer_id', 't_dat'])['article_id'].apply(list)
    multi_item_baskets = same_day_purchases[same_day_purchases.apply(len) > 1]
    
    logger.info(f"\nBasket completion opportunities:")
    logger.info(f"  Multi-item baskets: {len(multi_item_baskets):,}")
    logger.info(f"  Average items per multi-item basket: {multi_item_baskets.apply(len).mean():.2f}")
    
    return sparse_users, active_users, new_products


def generate_recommendations_summary():
    """Generate final recommendations for system design."""
    logger.info("\n\n" + "="*80)
    logger.info("RECOMMENDATION SYSTEM DESIGN GUIDELINES")
    logger.info("="*80)
    
    logger.info("""
Based on the analysis, here are the key recommendations:

1. MULTI-STAGE ARCHITECTURE
   Stage 1: Candidate Generation (1000s of items)
   - Popularity-based retrieval
   - Content-based similarity
   - Collaborative filtering for warm users
   
   Stage 2: Ranking (Top 12)
   - Deep learning models (NCF, Wide&Deep)
   - Feature-rich scoring
   - Personalization fine-tuning

2. COLD START STRATEGIES
   - New Users: Demographics → Popular items in age/gender group
   - New Items: Content features → Similar successful products
   - Fallback: Time-weighted global popularity

3. FEATURE ENGINEERING
   - User: Purchase history, diversity, frequency, recency
   - Item: Category, color, price, lifecycle stage
   - Context: Season, day of week, trends
   - Interaction: Co-purchase patterns, repeat buys

4. MODEL ENSEMBLE
   - Baseline: Popularity (strong baseline due to power law)
   - Short-term: Last N days purchases
   - Long-term: Collaborative filtering
   - Content: Item similarity
   - Temporal: Trend detection

5. EVALUATION STRATEGY
   - Time-based split (last week for test)
   - Cold start specific metrics
   - Diversity/novelty measures
   - A/B testing infrastructure

6. IMPLEMENTATION PRIORITIES
   Priority 1: Popularity baseline + basic content
   Priority 2: Collaborative filtering (ALS/BPR)
   Priority 3: Neural models (NCF)
   Priority 4: Temporal/sequential models
   Priority 5: Advanced graph models
""")
    
    logger.info("="*80)


def main():
    """Run complete recommendation challenges analysis."""
    logger.info("Starting recommendation challenges analysis...")
    
    # Run analyses
    customer_purchases, product_purchases = analyze_cold_start_problem()
    product_popularity, gini = analyze_popularity_bias()
    product_lifecycle, customer_frequency = analyze_temporal_dynamics()
    articles_df, customers_df = analyze_customer_item_features()
    sparse_users, active_users, new_products = analyze_recommendation_scenarios()
    
    # Generate summary
    generate_recommendations_summary()
    
    # Save analysis results
    results = {
        'cold_start_stats': {
            'customers_with_1_purchase': int((customer_purchases == 1).sum()),
            'products_with_1_purchase': int((product_purchases == 1).sum()),
        },
        'popularity_stats': {
            'gini_coefficient': float(gini),
            'top_1pct_purchase_share': float(product_popularity.head(int(len(product_popularity)*0.01)).sum() / product_popularity.sum())
        },
        'temporal_stats': {
            'avg_product_lifespan_days': float(product_lifecycle['lifespan_days'].mean()),
            'avg_customer_purchase_frequency_days': float(customer_frequency['purchase_frequency'].mean())
        }
    }
    
    import json
    results_path = project_root / 'experiments' / 'recommendation_challenges_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nAnalysis results saved to: {results_path}")


if __name__ == "__main__":
    main()