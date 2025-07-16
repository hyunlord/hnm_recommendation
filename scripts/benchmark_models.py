"""Quick benchmark script to evaluate model performance."""
import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.data import ImprovedHMDataModule
from src.models import (
    PopularityBaseline,
    MatrixFactorization,
    NeuralCF,
    WideDeep,
    LightGCN,
)
from src.evaluation import RecommendationMetrics
from src.utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark different recommendation models."""
    
    def __init__(
        self,
        data_dir: str = "data",
        sample_fraction: float = 0.01,
        batch_size: int = 1024,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize benchmark.
        
        Args:
            data_dir: Data directory
            sample_fraction: Fraction of data to use
            batch_size: Batch size for evaluation
            device: Device to use
        """
        self.data_dir = data_dir
        self.sample_fraction = sample_fraction
        self.batch_size = batch_size
        self.device = device
        
        # Initialize data module
        logger.info("Initializing data module...")
        self.data_module = ImprovedHMDataModule(
            data_dir=data_dir,
            sample_fraction=sample_fraction,
            batch_size=batch_size,
            num_workers=4,
            use_improved_datamodule=True,
        )
        self.data_module.setup()
        
        # Metrics
        self.metrics = RecommendationMetrics(top_k=12)
        
    def create_model(self, model_name: str) -> torch.nn.Module:
        """Create a model instance.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance
        """
        num_users = self.data_module.num_users
        num_items = self.data_module.num_items
        
        if model_name == "popularity_baseline":
            model = PopularityBaseline(
                num_items=num_items,
                top_k=12,
            )
            # Set popular items
            popular_items = self.data_module.get_popular_items(k=100)
            model.set_popular_items(popular_items)
            
        elif model_name == "matrix_factorization":
            model = MatrixFactorization(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=32,  # Smaller for quick test
                top_k=12,
            )
            
        elif model_name == "neural_cf":
            model = NeuralCF(
                num_users=num_users,
                num_items=num_items,
                mf_dim=32,
                mlp_dims=[64, 32, 16],  # Smaller for quick test
                dropout=0.1,
                top_k=12,
            )
            
        elif model_name == "wide_deep":
            model = WideDeep(
                num_users=num_users,
                num_items=num_items,
                num_user_features=self.data_module.num_user_features,
                num_item_features=self.data_module.num_item_features,
                embedding_dim=32,
                deep_layers=[128, 64, 32],  # Smaller for quick test
                dropout=0.1,
                top_k=12,
            )
            
        elif model_name == "lightgcn":
            model = LightGCN(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=32,
                num_layers=2,  # Fewer layers for quick test
                top_k=12,
            )
            # Set graph
            edge_index, edge_weight = self.data_module.get_graph()
            model.set_graph(edge_index.to(self.device), edge_weight)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model.to(self.device)
    
    def evaluate_model(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate a model.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader
            
        Returns:
            Metrics dictionary
        """
        model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                user_ids = batch['user_ids'].to(self.device)
                ground_truth = batch['ground_truth']
                
                # Get predictions
                if hasattr(model, 'predict_all_items'):
                    scores = model.predict_all_items(user_ids)
                else:
                    # For baseline model
                    scores = model(user_ids)
                
                # Get top-k items
                _, top_k_items = torch.topk(scores, 12, dim=1)
                
                # Update metrics
                self.metrics.update(top_k_items, ground_truth)
        
        return self.metrics.compute()
    
    def benchmark_all_models(self) -> pd.DataFrame:
        """Benchmark all models.
        
        Returns:
            DataFrame with results
        """
        models_to_test = [
            "popularity_baseline",
            "matrix_factorization",
            "neural_cf",
            "wide_deep",
            "lightgcn",
        ]
        
        results = []
        val_dataloader = self.data_module.val_dataloader()
        
        for model_name in models_to_test:
            logger.info(f"\\nBenchmarking {model_name}...")
            
            try:
                # Create model
                model = self.create_model(model_name)
                
                # Measure inference time
                start_time = time.time()
                metrics = self.evaluate_model(model, val_dataloader)
                inference_time = time.time() - start_time
                
                # Store results
                result = {
                    "model": model_name,
                    "map_at_k": metrics["map_at_k"],
                    "recall_at_k": metrics["recall_at_k"],
                    "precision_at_k": metrics["precision_at_k"],
                    "ndcg_at_k": metrics["ndcg_at_k"],
                    "inference_time": inference_time,
                    "status": "success"
                }
                
                logger.info(f"MAP@12: {metrics['map_at_k']:.4f}")
                logger.info(f"Inference time: {inference_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                result = {
                    "model": model_name,
                    "status": "failed",
                    "error": str(e)
                }
            
            results.append(result)
            
            # Clean up
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
        
        return pd.DataFrame(results)
    
    def save_results(self, results_df: pd.DataFrame, output_dir: str = "benchmark_results"):
        """Save benchmark results.
        
        Args:
            results_df: Results DataFrame
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_path = output_path / f"benchmark_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")
        
        # Save summary
        summary_path = output_path / f"benchmark_summary_{timestamp}.md"
        with open(summary_path, "w") as f:
            f.write("# Model Benchmark Results\\n\\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Sample Fraction**: {self.sample_fraction}\\n")
            f.write(f"**Device**: {self.device}\\n\\n")
            
            # Filter successful results
            success_df = results_df[results_df["status"] == "success"]
            
            if len(success_df) > 0:
                f.write("## Performance Summary\\n\\n")
                summary_cols = ["model", "map_at_k", "recall_at_k", "precision_at_k", "ndcg_at_k", "inference_time"]
                summary_df = success_df[summary_cols].round(4)
                summary_df = summary_df.sort_values("map_at_k", ascending=False)
                f.write(summary_df.to_markdown(index=False))
                
                f.write("\\n\\n## Best Model\\n\\n")
                best = summary_df.iloc[0]
                f.write(f"**{best['model']}** achieved the best MAP@12: {best['map_at_k']:.4f}\\n")
        
        logger.info(f"Summary saved to: {summary_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark recommendation models")
    parser.add_argument("--sample-fraction", type=float, default=0.01,
                       help="Fraction of data to use (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size (default: 1024)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create benchmark
    benchmark = ModelBenchmark(
        sample_fraction=args.sample_fraction,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Run benchmark
    logger.info("Starting model benchmark...")
    results_df = benchmark.benchmark_all_models()
    
    # Save results
    benchmark.save_results(results_df)
    
    # Print summary
    print("\\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    
    success_df = results_df[results_df["status"] == "success"]
    if len(success_df) > 0:
        print("\\nModel Performance (sorted by MAP@12):")
        summary_cols = ["model", "map_at_k", "recall_at_k", "inference_time"]
        summary_df = success_df[summary_cols].round(4)
        summary_df = summary_df.sort_values("map_at_k", ascending=False)
        print(summary_df.to_string(index=False))
    
    print("\\nNote: These are random initialization results.")
    print("Train models properly for meaningful comparisons!")


if __name__ == "__main__":
    main()