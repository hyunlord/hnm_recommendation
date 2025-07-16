"""Comprehensive analysis and visualization of experiment results."""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ExperimentAnalyzer:
    """Analyze and visualize experiment results."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        """Initialize analyzer.
        
        Args:
            experiments_dir: Directory containing experiment results
        """
        self.experiments_dir = Path(experiments_dir)
        self.results_df = None
        self.training_logs = {}
        
    def load_all_results(self) -> pd.DataFrame:
        """Load all experiment results.
        
        Returns:
            DataFrame with all results
        """
        all_results = []
        
        # Find all experiment directories
        for exp_dir in self.experiments_dir.glob("*"):
            if exp_dir.is_dir():
                # Load results from each model directory
                for model_dir in exp_dir.glob("*"):
                    if model_dir.is_dir():
                        result = self._load_single_result(model_dir)
                        if result:
                            result['experiment'] = exp_dir.name
                            result['model_dir'] = str(model_dir)
                            all_results.append(result)
        
        self.results_df = pd.DataFrame(all_results)
        
        # Clean and process results
        if len(self.results_df) > 0:
            self._process_results()
        
        return self.results_df
    
    def _load_single_result(self, model_dir: Path) -> Optional[Dict]:
        """Load result from a single model directory.
        
        Args:
            model_dir: Model directory path
            
        Returns:
            Result dictionary or None
        """
        result = {'model': model_dir.name}
        
        # Try to load results.yaml
        results_files = list(model_dir.glob("*_results.yaml"))
        if results_files:
            with open(results_files[0], 'r') as f:
                data = yaml.safe_load(f)
                result.update(data)
        
        # Try to load results.json
        elif (model_dir / "results.json").exists():
            with open(model_dir / "results.json", 'r') as f:
                data = json.load(f)
                result.update(data)
        
        # Load training logs if available
        log_files = list(model_dir.glob("logs/**/metrics.csv"))
        if log_files:
            self.training_logs[model_dir.name] = pd.read_csv(log_files[0])
        
        return result if 'test_map' in result or 'test_map_at_k' in result else None
    
    def _process_results(self):
        """Process and clean results DataFrame."""
        # Standardize column names
        if 'test_map' in self.results_df.columns:
            self.results_df['map_at_k'] = self.results_df['test_map']
        elif 'test_map_at_k' in self.results_df.columns:
            self.results_df['map_at_k'] = self.results_df['test_map_at_k']
        
        # Extract model type
        self.results_df['model_type'] = self.results_df['model'].apply(
            lambda x: x.split('_')[0] if '_' in x else x
        )
        
        # Add timestamp if available
        self.results_df['timestamp'] = pd.to_datetime(
            self.results_df['experiment'].str.extract(r'(\d{8}_\d{6})')[0],
            format='%Y%m%d_%H%M%S',
            errors='coerce'
        )
    
    def create_performance_dashboard(self, output_path: str = "results/dashboard.html"):
        """Create interactive performance dashboard.
        
        Args:
            output_path: Path to save dashboard HTML
        """
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to visualize")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 
                          'Performance by Model Type',
                          'Training Time vs Performance',
                          'Metric Correlations'),
            specs=[[{'type': 'bar'}, {'type': 'box'}],
                   [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        # 1. Model Performance Bar Chart
        perf_data = self.results_df.sort_values('map_at_k', ascending=False)
        fig.add_trace(
            go.Bar(
                x=perf_data['model'],
                y=perf_data['map_at_k'],
                name='MAP@12',
                text=perf_data['map_at_k'].round(4),
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # 2. Performance by Model Type (Box Plot)
        for model_type in self.results_df['model_type'].unique():
            data = self.results_df[self.results_df['model_type'] == model_type]
            fig.add_trace(
                go.Box(
                    y=data['map_at_k'],
                    name=model_type,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=1, col=2
            )
        
        # 3. Training Time vs Performance
        if 'duration' in self.results_df.columns:
            self.results_df['duration_min'] = self.results_df['duration'] / 60
            fig.add_trace(
                go.Scatter(
                    x=self.results_df['duration_min'],
                    y=self.results_df['map_at_k'],
                    mode='markers+text',
                    text=self.results_df['model'],
                    textposition='top center',
                    marker=dict(
                        size=10,
                        color=self.results_df['map_at_k'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Models'
                ),
                row=2, col=1
            )
        
        # 4. Metric Correlations Heatmap
        metrics = ['map_at_k', 'test_recall', 'test_precision', 'test_ndcg']
        available_metrics = [m for m in metrics if m in self.results_df.columns]
        
        if len(available_metrics) > 1:
            corr_matrix = self.results_df[available_metrics].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=available_metrics,
                    y=available_metrics,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(3),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="H&M Recommendation Models - Performance Dashboard",
            showlegend=True,
            height=1000,
            width=1400
        )
        
        # Update axes
        fig.update_xaxes(title_text="Model", row=1, col=1, tickangle=-45)
        fig.update_yaxes(title_text="MAP@12", row=1, col=1)
        fig.update_yaxes(title_text="MAP@12", row=1, col=2)
        fig.update_xaxes(title_text="Training Time (minutes)", row=2, col=1)
        fig.update_yaxes(title_text="MAP@12", row=2, col=1)
        
        # Save dashboard
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Dashboard saved to: {output_path}")
    
    def create_static_plots(self, output_dir: str = "results/plots"):
        """Create static visualization plots.
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to visualize")
            return
        
        # 1. Overall Performance Comparison
        self._plot_performance_comparison(output_dir)
        
        # 2. Model Type Analysis
        self._plot_model_type_analysis(output_dir)
        
        # 3. Training Curves
        self._plot_training_curves(output_dir)
        
        # 4. Performance Radar Chart
        self._plot_radar_chart(output_dir)
        
        # 5. Improvement Analysis
        self._plot_improvement_analysis(output_dir)
        
        print(f"Static plots saved to: {output_dir}")
    
    def _plot_performance_comparison(self, output_dir: Path):
        """Plot overall performance comparison."""
        plt.figure(figsize=(14, 8))
        
        # Sort by MAP@12
        sorted_df = self.results_df.sort_values('map_at_k', ascending=True)
        
        # Create horizontal bar plot
        colors = sns.color_palette("viridis", len(sorted_df))
        bars = plt.barh(sorted_df['model'], sorted_df['map_at_k'], color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.0005, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center')
        
        plt.xlabel('MAP@12', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title('Model Performance Comparison - MAP@12', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Highlight best model
        best_idx = sorted_df['map_at_k'].idxmax()
        plt.axhline(y=sorted_df.index.get_loc(best_idx), color='red', 
                   linestyle='--', alpha=0.5, label='Best Model')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_type_analysis(self, output_dir: Path):
        """Plot analysis by model type."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Box plot by model type
        ax = axes[0, 0]
        self.results_df.boxplot(column='map_at_k', by='model_type', ax=ax)
        ax.set_title('Performance Distribution by Model Type')
        ax.set_xlabel('Model Type')
        ax.set_ylabel('MAP@12')
        
        # 2. Average performance by model type
        ax = axes[0, 1]
        avg_perf = self.results_df.groupby('model_type')['map_at_k'].agg(['mean', 'std'])
        avg_perf = avg_perf.sort_values('mean', ascending=False)
        
        x = range(len(avg_perf))
        ax.bar(x, avg_perf['mean'], yerr=avg_perf['std'], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(avg_perf.index, rotation=45)
        ax.set_title('Average Performance by Model Type')
        ax.set_ylabel('MAP@12')
        
        # 3. Performance vs other metrics
        ax = axes[1, 0]
        if 'test_recall' in self.results_df.columns:
            ax.scatter(self.results_df['test_recall'], self.results_df['map_at_k'])
            for idx, row in self.results_df.iterrows():
                ax.annotate(row['model_type'], 
                           (row['test_recall'], row['map_at_k']),
                           fontsize=8, alpha=0.7)
            ax.set_xlabel('Recall@12')
            ax.set_ylabel('MAP@12')
            ax.set_title('MAP vs Recall')
        
        # 4. Count of experiments by model type
        ax = axes[1, 1]
        model_counts = self.results_df['model_type'].value_counts()
        ax.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
        ax.set_title('Distribution of Experiments by Model Type')
        
        plt.suptitle('Model Type Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'model_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_curves(self, output_dir: Path):
        """Plot training curves if available."""
        if not self.training_logs:
            return
        
        n_models = len(self.training_logs)
        fig, axes = plt.subplots(
            (n_models + 1) // 2, 2, 
            figsize=(15, 5 * ((n_models + 1) // 2))
        )
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, log_df) in enumerate(self.training_logs.items()):
            ax = axes[idx]
            
            # Plot training and validation loss
            if 'train_loss' in log_df.columns:
                ax.plot(log_df.index, log_df['train_loss'], label='Train Loss', alpha=0.8)
            if 'val_loss' in log_df.columns:
                ax.plot(log_df.index, log_df['val_loss'], label='Val Loss', alpha=0.8)
            if 'val_map_at_k' in log_df.columns:
                ax2 = ax.twinx()
                ax2.plot(log_df.index, log_df['val_map_at_k'], 
                        label='Val MAP@12', color='green', alpha=0.8)
                ax2.set_ylabel('MAP@12', color='green')
                ax2.tick_params(axis='y', labelcolor='green')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training Curves - {model_name}')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(self.training_logs), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_radar_chart(self, output_dir: Path):
        """Plot radar chart comparing top models across metrics."""
        metrics = ['map_at_k', 'test_recall', 'test_precision', 'test_ndcg']
        available_metrics = [m for m in metrics if m in self.results_df.columns]
        
        if len(available_metrics) < 3:
            return
        
        # Get top 5 models
        top_models = self.results_df.nlargest(5, 'map_at_k')
        
        # Prepare data
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for idx, (_, model) in enumerate(top_models.iterrows()):
            values = [model[metric] for metric in available_metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model['model'])
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('test_', '').replace('_', ' ').title() 
                           for m in available_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Top 5 Models - Multi-Metric Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_analysis(self, output_dir: Path):
        """Plot improvement analysis over baseline."""
        if 'popularity_baseline' not in self.results_df['model'].values:
            return
        
        baseline_map = self.results_df[
            self.results_df['model'] == 'popularity_baseline'
        ]['map_at_k'].values[0]
        
        # Calculate improvements
        self.results_df['improvement'] = (
            (self.results_df['map_at_k'] - baseline_map) / baseline_map * 100
        )
        
        # Sort by improvement
        sorted_df = self.results_df[
            self.results_df['model'] != 'popularity_baseline'
        ].sort_values('improvement', ascending=True)
        
        plt.figure(figsize=(12, 8))
        
        # Color based on positive/negative improvement
        colors = ['green' if x > 0 else 'red' for x in sorted_df['improvement']]
        
        bars = plt.barh(sorted_df['model'], sorted_df['improvement'], color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label = f'{width:+.1f}%'
            plt.text(width + 0.5 if width > 0 else width - 0.5, 
                    bar.get_y() + bar.get_height()/2,
                    label, ha='left' if width > 0 else 'right', va='center')
        
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.xlabel('Improvement over Baseline (%)', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(f'Performance Improvement over Popularity Baseline (MAP@12={baseline_map:.4f})',
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_path: str = "results/experiment_report.md"):
        """Generate comprehensive markdown report.
        
        Args:
            output_path: Path to save report
        """
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to report")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Header
            f.write("# H&M Recommendation System - Experiment Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary Statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- Total experiments: {len(self.results_df)}\n")
            f.write(f"- Unique models: {self.results_df['model_type'].nunique()}\n")
            f.write(f"- Best MAP@12: {self.results_df['map_at_k'].max():.4f}\n")
            f.write(f"- Average MAP@12: {self.results_df['map_at_k'].mean():.4f}\n")
            f.write(f"- Std MAP@12: {self.results_df['map_at_k'].std():.4f}\n\n")
            
            # Best Model
            best_model = self.results_df.loc[self.results_df['map_at_k'].idxmax()]
            f.write("## Best Performing Model\n\n")
            f.write(f"**Model**: {best_model['model']}\n")
            f.write(f"**MAP@12**: {best_model['map_at_k']:.4f}\n")
            
            if 'test_recall' in best_model:
                f.write(f"**Recall@12**: {best_model['test_recall']:.4f}\n")
            if 'test_precision' in best_model:
                f.write(f"**Precision@12**: {best_model['test_precision']:.4f}\n")
            if 'test_ndcg' in best_model:
                f.write(f"**NDCG@12**: {best_model['test_ndcg']:.4f}\n")
            
            f.write("\n")
            
            # Performance Table
            f.write("## Performance Comparison\n\n")
            summary_cols = ['model', 'map_at_k', 'test_recall', 'test_precision', 'test_ndcg']
            available_cols = [col for col in summary_cols if col in self.results_df.columns]
            
            summary_df = self.results_df[available_cols].round(4)
            summary_df = summary_df.sort_values('map_at_k', ascending=False)
            
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n")
            
            # Model Type Analysis
            f.write("## Model Type Analysis\n\n")
            type_summary = self.results_df.groupby('model_type')['map_at_k'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4)
            type_summary = type_summary.sort_values('mean', ascending=False)
            
            f.write(type_summary.to_markdown())
            f.write("\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Top 3 models
            top_3 = self.results_df.nlargest(3, 'map_at_k')
            f.write("### Top 3 Models\n\n")
            for idx, (_, model) in enumerate(top_3.iterrows(), 1):
                f.write(f"{idx}. **{model['model']}**: MAP@12 = {model['map_at_k']:.4f}\n")
            
            f.write("\n### Next Steps\n\n")
            f.write("1. **Hyperparameter Tuning**: Focus on top performing models\n")
            f.write("2. **Ensemble Methods**: Combine predictions from top models\n")
            f.write("3. **Feature Engineering**: Add more user and item features\n")
            f.write("4. **Production Testing**: Deploy best model with A/B testing\n")
            f.write("5. **Cold Start Solutions**: Address new user/item challenges\n")
            
            # Technical Details
            f.write("\n## Technical Details\n\n")
            f.write("### Experiment Configuration\n\n")
            f.write("- Evaluation Metric: MAP@12 (Mean Average Precision)\n")
            f.write("- Train/Val/Test Split: Time-based (104/1/1 weeks)\n")
            f.write("- Negative Sampling: Various strategies tested\n")
            f.write("- Hardware: GPU/CPU as available\n")
            
        print(f"Report saved to: {output_path}")
    
    def create_model_comparison_matrix(self, output_path: str = "results/comparison_matrix.png"):
        """Create a detailed model comparison matrix.
        
        Args:
            output_path: Path to save the comparison matrix
        """
        if self.results_df is None or len(self.results_df) == 0:
            return
        
        # Metrics to compare
        metrics = ['map_at_k', 'test_recall', 'test_precision', 'test_ndcg']
        available_metrics = [m for m in metrics if m in self.results_df.columns]
        
        if 'duration' in self.results_df.columns:
            self.results_df['efficiency'] = (
                self.results_df['map_at_k'] / (self.results_df['duration'] / 3600)
            )
            available_metrics.append('efficiency')
        
        # Create comparison matrix
        comparison_df = self.results_df.set_index('model')[available_metrics]
        
        # Normalize to 0-1 scale
        normalized_df = (comparison_df - comparison_df.min()) / (comparison_df.max() - comparison_df.min())
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Create custom colormap
        colors = ['#d73027', '#fee08b', '#1a9850']
        n_bins = 100
        cmap = sns.blend_palette(colors, n_colors=n_bins, as_cmap=True)
        
        # Plot heatmap
        sns.heatmap(
            normalized_df.T,
            annot=comparison_df.T.round(3),
            fmt='.3f',
            cmap=cmap,
            cbar_kws={'label': 'Normalized Score'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title('Model Comparison Matrix (Normalized Scores)', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        
        # Add metric labels
        metric_labels = {
            'map_at_k': 'MAP@12',
            'test_recall': 'Recall@12',
            'test_precision': 'Precision@12',
            'test_ndcg': 'NDCG@12',
            'efficiency': 'MAP/Hour'
        }
        
        y_labels = [metric_labels.get(m, m) for m in available_metrics]
        plt.gca().set_yticklabels(y_labels)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison matrix saved to: {output_path}")


def main():
    """Main function to run analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                       help="Directory containing experiments")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--create-dashboard", action="store_true",
                       help="Create interactive dashboard")
    parser.add_argument("--create-plots", action="store_true",
                       help="Create static plots")
    parser.add_argument("--create-report", action="store_true",
                       help="Generate markdown report")
    parser.add_argument("--all", action="store_true",
                       help="Create all outputs")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ExperimentAnalyzer(args.experiments_dir)
    
    # Load results
    print("Loading experiment results...")
    results_df = analyzer.load_all_results()
    
    if len(results_df) == 0:
        print("No results found!")
        return
    
    print(f"Loaded {len(results_df)} experiment results")
    
    # Create outputs
    if args.all or args.create_dashboard:
        print("\nCreating interactive dashboard...")
        analyzer.create_performance_dashboard(f"{args.output_dir}/dashboard.html")
    
    if args.all or args.create_plots:
        print("\nCreating static plots...")
        analyzer.create_static_plots(f"{args.output_dir}/plots")
        analyzer.create_model_comparison_matrix(f"{args.output_dir}/comparison_matrix.png")
    
    if args.all or args.create_report:
        print("\nGenerating report...")
        analyzer.generate_report(f"{args.output_dir}/experiment_report.md")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nTop 5 Models by MAP@12:")
    top_5 = results_df.nlargest(5, 'map_at_k')[['model', 'map_at_k']]
    print(top_5.to_string(index=False))
    
    print(f"\nOutputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()