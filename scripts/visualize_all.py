"""Comprehensive visualization script that generates all analysis outputs."""
import os
import sys
from pathlib import Path
import subprocess
import logging
from datetime import datetime
import webbrowser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_analysis_pipeline(experiments_dir: str = "experiments", output_dir: str = "results"):
    """Run complete analysis pipeline.
    
    Args:
        experiments_dir: Directory containing experiments
        output_dir: Directory to save outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_path / f"analysis_{timestamp}"
    session_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting analysis pipeline...")
    logger.info(f"Experiments directory: {experiments_dir}")
    logger.info(f"Output directory: {session_dir}")
    
    # 1. Run comprehensive analysis
    logger.info("\n1. Running comprehensive analysis...")
    try:
        subprocess.run([
            sys.executable, "scripts/analyze_results.py",
            "--experiments-dir", experiments_dir,
            "--output-dir", str(session_dir),
            "--all"
        ], check=True)
        logger.info("✅ Analysis completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Analysis failed: {e}")
        return
    
    # 2. Generate experiment report
    logger.info("\n2. Generating detailed report...")
    report_path = session_dir / "experiment_report.md"
    if report_path.exists():
        logger.info(f"✅ Report generated: {report_path}")
    
    # 3. Create summary visualization
    logger.info("\n3. Creating summary visualizations...")
    create_summary_page(session_dir)
    
    # 4. Launch Streamlit dashboard (optional)
    logger.info("\n4. Dashboard ready to launch")
    logger.info("To view interactive dashboard, run:")
    logger.info(f"  streamlit run scripts/dashboard_app.py")
    
    # 5. Open results in browser
    summary_path = session_dir / "summary.html"
    if summary_path.exists():
        logger.info(f"\n5. Opening results in browser...")
        webbrowser.open(f"file://{summary_path.absolute()}")
    
    logger.info(f"\n✅ All visualizations completed!")
    logger.info(f"📁 Results saved to: {session_dir}")
    
    return session_dir


def create_summary_page(output_dir: Path):
    """Create an HTML summary page linking all outputs.
    
    Args:
        output_dir: Directory containing analysis outputs
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>H&M Recommendation - Experiment Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #1f77b4;
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0;
            font-size: 2.5rem;
        }}
        .subtitle {{
            margin-top: 0.5rem;
            opacity: 0.9;
        }}
        .section {{
            background-color: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #1f77b4;
            margin-top: 0;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}
        .card {{
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin-top: 0;
            color: #495057;
        }}
        .card a {{
            display: inline-block;
            margin-top: 1rem;
            padding: 0.5rem 1rem;
            background-color: #1f77b4;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.2s;
        }}
        .card a:hover {{
            background-color: #155d8f;
        }}
        .metrics {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1.5rem;
        }}
        .metric {{
            text-align: center;
            padding: 1rem;
            background-color: #e7f3ff;
            border-radius: 8px;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #1f77b4;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        .plot-container {{
            margin-top: 1.5rem;
        }}
        .plot-container img {{
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            text-align: right;
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛍️ H&M Recommendation System</h1>
        <p class="subtitle">Experiment Results and Analysis</p>
    </div>
    
    <div class="section">
        <h2>📊 Quick Overview</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">-</div>
                <div class="metric-label">Total Experiments</div>
            </div>
            <div class="metric">
                <div class="metric-value">-</div>
                <div class="metric-label">Best MAP@12</div>
            </div>
            <div class="metric">
                <div class="metric-value">-</div>
                <div class="metric-label">Best Model</div>
            </div>
            <div class="metric">
                <div class="metric-value">-</div>
                <div class="metric-label">Avg Performance</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Analysis Outputs</h2>
        <div class="grid">
            <div class="card">
                <h3>📊 Interactive Dashboard</h3>
                <p>Explore experiment results with interactive visualizations and filtering.</p>
                <a href="dashboard.html" target="_blank">Open Dashboard</a>
            </div>
            
            <div class="card">
                <h3>📋 Detailed Report</h3>
                <p>Comprehensive markdown report with all metrics and recommendations.</p>
                <a href="experiment_report.md" target="_blank">View Report</a>
            </div>
            
            <div class="card">
                <h3>🖼️ Static Plots</h3>
                <p>Collection of publication-ready plots and visualizations.</p>
                <a href="plots/" target="_blank">Browse Plots</a>
            </div>
            
            <div class="card">
                <h3>📊 Comparison Matrix</h3>
                <p>Detailed model comparison across all metrics.</p>
                <a href="comparison_matrix.png" target="_blank">View Matrix</a>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🏆 Key Visualizations</h2>
        <div class="plot-container">
            <img src="plots/performance_comparison.png" alt="Performance Comparison" 
                 onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"400\"%3E%3Crect width=\"800\" height=\"400\" fill=\"%23f0f0f0\"/%3E%3Ctext x=\"400\" y=\"200\" font-family=\"Arial\" font-size=\"20\" fill=\"%23999\" text-anchor=\"middle\"%3EPerformance Comparison Plot%3C/text%3E%3C/svg%3E';">
        </div>
    </div>
    
    <div class="section">
        <h2>🚀 Next Steps</h2>
        <ol>
            <li><strong>Review the detailed report</strong> for comprehensive analysis</li>
            <li><strong>Explore the interactive dashboard</strong> for dynamic insights</li>
            <li><strong>Run hyperparameter tuning</strong> on the best performing models</li>
            <li><strong>Deploy the best model</strong> using the API server</li>
            <li><strong>Set up A/B testing</strong> for production validation</li>
        </ol>
    </div>
    
    <div class="timestamp">
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
"""
    
    summary_path = output_dir / "summary.html"
    with open(summary_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"✅ Summary page created: {summary_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize all experiment results")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                       help="Directory containing experiments")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save outputs")
    parser.add_argument("--launch-dashboard", action="store_true",
                       help="Launch Streamlit dashboard after analysis")
    
    args = parser.parse_args()
    
    # Run analysis pipeline
    session_dir = run_analysis_pipeline(args.experiments_dir, args.output_dir)
    
    # Launch dashboard if requested
    if args.launch_dashboard:
        logger.info("\nLaunching Streamlit dashboard...")
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "scripts/dashboard_app.py"
        ])
        logger.info("Dashboard launched! Opening in browser...")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {session_dir}")
    print("\nTo view results:")
    print(f"1. Open summary: {session_dir}/summary.html")
    print("2. Run dashboard: streamlit run scripts/dashboard_app.py")
    print("3. View plots: {}/plots/".format(session_dir))


if __name__ == "__main__":
    main()