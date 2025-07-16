"""Script to run comprehensive model comparison experiments."""
import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Manages and runs model comparison experiments."""
    
    def __init__(
        self, 
        base_dir: str = "experiments",
        sample_fraction: float = 0.1,
        quick_test: bool = False
    ):
        """Initialize experiment runner.
        
        Args:
            base_dir: Base directory for experiment outputs
            sample_fraction: Fraction of data to use (for faster experiments)
            quick_test: If True, run quick tests with fewer epochs
        """
        self.base_dir = Path(base_dir)
        self.sample_fraction = sample_fraction
        self.quick_test = quick_test
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / f"comparison_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Define experiments
        self.experiments = self._define_experiments()
        
    def _define_experiments(self) -> List[Dict]:
        """Define all experiments to run."""
        experiments = []
        
        # Base configuration
        base_config = {
            "data.sample_fraction": self.sample_fraction,
            "training.num_workers": 4,
            "logging.enabled": True,
        }
        
        if self.quick_test:
            base_config["training.epochs"] = 5
            base_config["training.val_check_interval"] = 1.0
        
        # 1. Popularity Baseline
        experiments.append({
            "name": "popularity_baseline",
            "model": "popularity_baseline",
            "config": {**base_config}
        })
        
        # 2. Matrix Factorization
        experiments.append({
            "name": "matrix_factorization",
            "model": "matrix_factorization",
            "config": {
                **base_config,
                "model.embedding_dim": 64,
                "model.learning_rate": 0.001,
                "data.negative_sampling_ratio": 4,
            }
        })
        
        # 3. Neural Collaborative Filtering
        experiments.append({
            "name": "neural_cf",
            "model": "neural_cf",
            "config": {
                **base_config,
                "model.mf_dim": 64,
                "model.mlp_dims": "[128,64,32]",
                "model.dropout": 0.1,
                "data.negative_sampling_ratio": 4,
            }
        })
        
        # 4. Wide & Deep
        experiments.append({
            "name": "wide_deep",
            "model": "wide_deep",
            "config": {
                **base_config,
                "model.deep_layers": "[512,256,128]",
                "model.dropout": 0.1,
                "data.use_features": True,
                "data.negative_sampling_ratio": 4,
            }
        })
        
        # 5. LightGCN
        experiments.append({
            "name": "lightgcn",
            "model": "lightgcn",
            "config": {
                **base_config,
                "model.embedding_dim": 64,
                "model.num_layers": 3,
                "data.dataset_type": "bpr",
                "training.batch_size": 2048,
            }
        })
        
        # 6. Advanced experiments with different sampling strategies
        if not self.quick_test:
            # Neural CF with popularity sampling
            experiments.append({
                "name": "neural_cf_popularity",
                "model": "neural_cf",
                "config": {
                    **base_config,
                    "data.sampling_strategy": "popularity",
                    "data.negative_sampling_ratio": 4,
                }
            })
            
            # Neural CF with hard negative sampling
            experiments.append({
                "name": "neural_cf_hard",
                "model": "neural_cf",
                "config": {
                    **base_config,
                    "data.sampling_strategy": "hard",
                    "data.negative_sampling_ratio": 4,
                }
            })
            
            # Wide & Deep without features
            experiments.append({
                "name": "wide_deep_no_features",
                "model": "wide_deep",
                "config": {
                    **base_config,
                    "data.use_features": False,
                    "data.negative_sampling_ratio": 4,
                }
            })
        
        return experiments
    
    def run_experiment(self, experiment: Dict) -> Dict:
        """Run a single experiment.
        
        Args:
            experiment: Experiment configuration
            
        Returns:
            Results dictionary
        """
        name = experiment["name"]
        logger.info(f"Running experiment: {name}")
        
        # Create experiment directory
        exp_dir = self.experiment_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        # Build command
        cmd = ["python", "scripts/train.py"]
        cmd.append(f"model={experiment['model']}")
        
        # Add configuration overrides
        for key, value in experiment["config"].items():
            cmd.append(f"{key}={value}")
        
        # Add output directory
        cmd.append(f"paths.output_dir={exp_dir}")
        cmd.append(f"run_name={name}")
        
        # Run experiment
        logger.info(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            duration = time.time() - start_time
            
            # Parse results from output
            results = self._parse_results(result.stdout, exp_dir)
            results["duration"] = duration
            results["status"] = "success"
            results["name"] = name
            results["model"] = experiment["model"]
            
            logger.info(f"Experiment {name} completed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Experiment {name} failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            results = {
                "name": name,
                "model": experiment["model"],
                "status": "failed",
                "error": str(e),
                "duration": time.time() - start_time
            }
        
        # Save individual experiment results
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _parse_results(self, output: str, exp_dir: Path) -> Dict:
        """Parse results from training output.
        
        Args:
            output: Training script output
            exp_dir: Experiment directory
            
        Returns:
            Parsed results
        """
        results = {}
        
        # Try to load results from saved YAML file
        results_files = list(exp_dir.glob("*_results.yaml"))
        if results_files:
            with open(results_files[0], "r") as f:
                saved_results = yaml.safe_load(f)
                results.update(saved_results)
        
        # Parse from output as backup
        lines = output.split("\n")
        for line in lines:
            if "test_map_at_k" in line:
                try:
                    value = float(line.split(":")[-1].strip())
                    results["test_map_at_k"] = value
                except:
                    pass
            elif "test_recall_at_k" in line:
                try:
                    value = float(line.split(":")[-1].strip())
                    results["test_recall_at_k"] = value
                except:
                    pass
            elif "test_ndcg_at_k" in line:
                try:
                    value = float(line.split(":")[-1].strip())
                    results["test_ndcg_at_k"] = value
                except:
                    pass
        
        return results
    
    def run_all_experiments(self) -> pd.DataFrame:
        """Run all defined experiments.
        
        Returns:
            DataFrame with all results
        """
        logger.info(f"Starting {len(self.experiments)} experiments")
        logger.info(f"Results will be saved to: {self.experiment_dir}")
        
        all_results = []
        
        for i, experiment in enumerate(self.experiments):
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {i+1}/{len(self.experiments)}: {experiment['name']}")
            logger.info(f"{'='*60}")
            
            results = self.run_experiment(experiment)
            all_results.append(results)
            
            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv(self.experiment_dir / "results_intermediate.csv", index=False)
        
        # Create final results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.experiment_dir / "results_final.csv", index=False)
        
        logger.info("\nAll experiments completed!")
        return results_df
    
    def create_visualizations(self, results_df: pd.DataFrame):
        """Create visualization plots for experiment results.
        
        Args:
            results_df: DataFrame with experiment results
        """
        logger.info("Creating visualizations...")
        
        # Filter successful experiments
        success_df = results_df[results_df["status"] == "success"].copy()
        
        if len(success_df) == 0:
            logger.warning("No successful experiments to visualize")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Overall performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ["test_map_at_k", "test_recall_at_k", "test_precision", "test_ndcg_at_k"]
        titles = ["MAP@12", "Recall@12", "Precision@12", "NDCG@12"]
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            if metric in success_df.columns:
                data = success_df[["name", metric]].dropna()
                if len(data) > 0:
                    ax.bar(data["name"], data[metric])
                    ax.set_title(f"{title} Comparison", fontsize=14)
                    ax.set_xlabel("Model", fontsize=12)
                    ax.set_ylabel(title, fontsize=12)
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels
                    for i, v in enumerate(data[metric]):
                        ax.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "performance_comparison.png", dpi=300)
        plt.close()
        
        # 2. Training time comparison
        if "duration" in success_df.columns:
            plt.figure(figsize=(12, 6))
            success_df["duration_min"] = success_df["duration"] / 60
            ax = sns.barplot(data=success_df, x="name", y="duration_min")
            plt.title("Training Time Comparison", fontsize=14)
            plt.xlabel("Model", fontsize=12)
            plt.ylabel("Training Time (minutes)", fontsize=12)
            plt.xticks(rotation=45)
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}',
                           (p.get_x() + p.get_width()/2., p.get_height()),
                           ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.experiment_dir / "training_time_comparison.png", dpi=300)
            plt.close()
        
        # 3. Create summary table
        if "test_map_at_k" in success_df.columns:
            summary_metrics = ["test_map_at_k", "test_recall_at_k", "test_ndcg_at_k", "duration_min"]
            summary_df = success_df[["name", "model"] + 
                                  [m for m in summary_metrics if m in success_df.columns]]
            
            # Round numerical values
            for col in summary_metrics:
                if col in summary_df.columns:
                    if col == "duration_min":
                        summary_df[col] = summary_df[col].round(1)
                    else:
                        summary_df[col] = summary_df[col].round(4)
            
            # Sort by MAP@12
            if "test_map_at_k" in summary_df.columns:
                summary_df = summary_df.sort_values("test_map_at_k", ascending=False)
            
            # Save as markdown table
            with open(self.experiment_dir / "summary_table.md", "w") as f:
                f.write("# Model Performance Summary\n\n")
                f.write(summary_df.to_markdown(index=False))
        
        logger.info(f"Visualizations saved to: {self.experiment_dir}")
    
    def create_report(self, results_df: pd.DataFrame):
        """Create a comprehensive experiment report.
        
        Args:
            results_df: DataFrame with experiment results
        """
        logger.info("Creating experiment report...")
        
        report_path = self.experiment_dir / "experiment_report.md"
        
        with open(report_path, "w") as f:
            f.write(f"# H&M Recommendation Model Comparison Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Sample Fraction**: {self.sample_fraction}\n")
            f.write(f"**Quick Test**: {self.quick_test}\n\n")
            
            # Summary statistics
            success_df = results_df[results_df["status"] == "success"]
            f.write(f"## Summary\n\n")
            f.write(f"- Total experiments: {len(results_df)}\n")
            f.write(f"- Successful: {len(success_df)}\n")
            f.write(f"- Failed: {len(results_df) - len(success_df)}\n\n")
            
            # Best performing model
            if len(success_df) > 0 and "test_map_at_k" in success_df.columns:
                best_model = success_df.loc[success_df["test_map_at_k"].idxmax()]
                f.write(f"## Best Performing Model\n\n")
                f.write(f"**{best_model['name']}** achieved the highest MAP@12: {best_model['test_map_at_k']:.4f}\n\n")
            
            # Performance table
            if len(success_df) > 0:
                f.write(f"## Performance Comparison\n\n")
                summary_cols = ["name", "test_map_at_k", "test_recall_at_k", "test_ndcg_at_k"]
                summary_cols = [col for col in summary_cols if col in success_df.columns]
                
                if summary_cols:
                    summary_df = success_df[summary_cols].round(4)
                    if "test_map_at_k" in summary_df.columns:
                        summary_df = summary_df.sort_values("test_map_at_k", ascending=False)
                    f.write(summary_df.to_markdown(index=False))
                    f.write("\n\n")
            
            # Failed experiments
            failed_df = results_df[results_df["status"] == "failed"]
            if len(failed_df) > 0:
                f.write(f"## Failed Experiments\n\n")
                for _, row in failed_df.iterrows():
                    f.write(f"- **{row['name']}**: {row.get('error', 'Unknown error')}\n")
                f.write("\n")
            
            # Recommendations
            f.write(f"## Recommendations\n\n")
            if len(success_df) > 0 and "test_map_at_k" in success_df.columns:
                top_models = success_df.nlargest(3, "test_map_at_k")
                f.write("Based on the experiments, the top 3 models are:\n\n")
                for i, (_, model) in enumerate(top_models.iterrows(), 1):
                    f.write(f"{i}. **{model['name']}**: MAP@12 = {model['test_map_at_k']:.4f}\n")
            
        logger.info(f"Report saved to: {report_path}")


def main():
    """Main function to run experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run H&M recommendation model experiments")
    parser.add_argument("--sample-fraction", type=float, default=0.1,
                       help="Fraction of data to use (default: 0.1)")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick tests with fewer epochs")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                       help="Base directory for experiments (default: experiments)")
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(
        base_dir=args.experiments_dir,
        sample_fraction=args.sample_fraction,
        quick_test=args.quick_test
    )
    
    # Run all experiments
    results_df = runner.run_all_experiments()
    
    # Create visualizations and report
    runner.create_visualizations(results_df)
    runner.create_report(results_df)
    
    logger.info("\nExperiment comparison completed!")
    logger.info(f"Results saved to: {runner.experiment_dir}")


if __name__ == "__main__":
    main()