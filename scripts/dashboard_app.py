"""Interactive dashboard app for experiment monitoring using Streamlit."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import yaml
import json
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="H&M Recommendation Dashboard",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_experiment_results(experiments_dir: Path) -> pd.DataFrame:
    """Load all experiment results with caching."""
    all_results = []
    
    for exp_dir in experiments_dir.glob("*"):
        if exp_dir.is_dir():
            for model_dir in exp_dir.glob("*"):
                if model_dir.is_dir():
                    result = load_single_result(model_dir)
                    if result:
                        result['experiment'] = exp_dir.name
                        result['model_path'] = str(model_dir)
                        all_results.append(result)
    
    df = pd.DataFrame(all_results)
    
    # Process results
    if len(df) > 0:
        if 'test_map' in df.columns:
            df['MAP@12'] = df['test_map']
        elif 'test_map_at_k' in df.columns:
            df['MAP@12'] = df['test_map_at_k']
        
        # Extract timestamp
        df['timestamp'] = pd.to_datetime(
            df['experiment'].str.extract(r'(\d{8}_\d{6})')[0],
            format='%Y%m%d_%H%M%S',
            errors='coerce'
        )
        
        # Model type
        df['model_type'] = df['model'].apply(
            lambda x: x.split('_')[0] if '_' in x else x
        )
    
    return df


def load_single_result(model_dir: Path) -> dict:
    """Load result from a single model directory."""
    result = {'model': model_dir.name}
    
    # Try different result file formats
    results_files = list(model_dir.glob("*_results.yaml"))
    if results_files:
        with open(results_files[0], 'r') as f:
            data = yaml.safe_load(f)
            result.update(data)
    elif (model_dir / "results.json").exists():
        with open(model_dir / "results.json", 'r') as f:
            data = json.load(f)
            result.update(data)
    
    return result if any(k in result for k in ['test_map', 'test_map_at_k']) else None


def main():
    """Main dashboard application."""
    st.title("🛍️ H&M Recommendation System Dashboard")
    st.markdown("Real-time monitoring and analysis of recommendation model experiments")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Experiment directory
        experiments_dir = st.text_input(
            "Experiments Directory",
            value="experiments",
            help="Path to experiments directory"
        )
        experiments_path = Path(experiments_dir)
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
        if auto_refresh:
            time.sleep(60)
            st.experimental_rerun()
    
    # Load data
    if not experiments_path.exists():
        st.error(f"Experiments directory not found: {experiments_path}")
        return
    
    with st.spinner("Loading experiment results..."):
        results_df = load_experiment_results(experiments_path)
    
    if len(results_df) == 0:
        st.warning("No experiment results found!")
        st.info("Run some experiments first using `python scripts/train.py` or `python scripts/run_experiments.py`")
        return
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Performance Analysis", 
        "🔬 Model Comparison",
        "📉 Training Progress",
        "📋 Detailed Results"
    ])
    
    with tab1:
        display_overview(results_df)
    
    with tab2:
        display_performance_analysis(results_df)
    
    with tab3:
        display_model_comparison(results_df)
    
    with tab4:
        display_training_progress(results_df, experiments_path)
    
    with tab5:
        display_detailed_results(results_df)


def display_overview(df: pd.DataFrame):
    """Display overview metrics and statistics."""
    st.header("Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Experiments",
            len(df),
            delta=None
        )
    
    with col2:
        st.metric(
            "Best MAP@12",
            f"{df['MAP@12'].max():.4f}" if 'MAP@12' in df.columns else "N/A",
            delta=None
        )
    
    with col3:
        st.metric(
            "Unique Models",
            df['model_type'].nunique() if 'model_type' in df.columns else 0,
            delta=None
        )
    
    with col4:
        latest_exp = df['timestamp'].max() if 'timestamp' in df.columns else None
        st.metric(
            "Latest Experiment",
            latest_exp.strftime("%Y-%m-%d %H:%M") if pd.notna(latest_exp) else "N/A",
            delta=None
        )
    
    # Best performing models
    st.subheader("🏆 Top Performing Models")
    
    if 'MAP@12' in df.columns:
        top_models = df.nlargest(5, 'MAP@12')[['model', 'MAP@12', 'model_type']]
        
        # Create bar chart
        fig = px.bar(
            top_models,
            x='MAP@12',
            y='model',
            orientation='h',
            color='model_type',
            title="Top 5 Models by MAP@12",
            labels={'MAP@12': 'MAP@12', 'model': 'Model'},
            text='MAP@12'
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent experiments
    st.subheader("📅 Recent Experiments")
    if 'timestamp' in df.columns:
        recent = df.nlargest(10, 'timestamp')[['timestamp', 'model', 'MAP@12']]
        recent['timestamp'] = recent['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent, use_container_width=True)


def display_performance_analysis(df: pd.DataFrame):
    """Display performance analysis visualizations."""
    st.header("Performance Analysis")
    
    if 'MAP@12' not in df.columns:
        st.warning("No performance metrics found")
        return
    
    # Model type comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot by model type
        fig = px.box(
            df,
            x='model_type',
            y='MAP@12',
            title="Performance Distribution by Model Type",
            points="all"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average performance by model type
        avg_perf = df.groupby('model_type')['MAP@12'].agg(['mean', 'std', 'count']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=avg_perf['model_type'],
            y=avg_perf['mean'],
            error_y=dict(type='data', array=avg_perf['std']),
            text=avg_perf['count'],
            texttemplate='n=%{text}',
            textposition='outside',
            name='Mean MAP@12'
        ))
        fig.update_layout(
            title="Average Performance by Model Type",
            xaxis_title="Model Type",
            yaxis_title="MAP@12",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance over time
    if 'timestamp' in df.columns and pd.notna(df['timestamp']).any():
        st.subheader("Performance Over Time")
        
        # Group by date and model type
        df['date'] = df['timestamp'].dt.date
        time_perf = df.groupby(['date', 'model_type'])['MAP@12'].mean().reset_index()
        
        fig = px.line(
            time_perf,
            x='date',
            y='MAP@12',
            color='model_type',
            title="Performance Trends Over Time",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Metric correlations
    metrics = ['MAP@12', 'test_recall', 'test_precision', 'test_ndcg']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) > 1:
        st.subheader("Metric Correlations")
        
        corr_matrix = df[available_metrics].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=available_metrics,
            y=available_metrics,
            color_continuous_scale='RdBu',
            aspect="auto",
            title="Correlation Matrix of Evaluation Metrics"
        )
        fig.update_traces(text=corr_matrix.values.round(3), texttemplate='%{text}')
        st.plotly_chart(fig, use_container_width=True)


def display_model_comparison(df: pd.DataFrame):
    """Display detailed model comparison."""
    st.header("Model Comparison")
    
    # Model selection
    available_models = df['model'].unique()
    selected_models = st.multiselect(
        "Select models to compare",
        available_models,
        default=list(df.nlargest(5, 'MAP@12')['model']) if 'MAP@12' in df.columns else []
    )
    
    if not selected_models:
        st.info("Select models to compare")
        return
    
    # Filter data
    comparison_df = df[df['model'].isin(selected_models)]
    
    # Radar chart
    metrics = ['MAP@12', 'test_recall', 'test_precision', 'test_ndcg']
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    if len(available_metrics) >= 3:
        st.subheader("Multi-Metric Comparison")
        
        fig = go.Figure()
        
        for model in selected_models:
            model_data = comparison_df[comparison_df['model'] == model].iloc[0]
            values = [model_data[m] for m in available_metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Side-by-side comparison
    st.subheader("Side-by-Side Comparison")
    
    comparison_metrics = ['MAP@12'] + [m for m in ['test_recall', 'test_precision', 'test_ndcg'] 
                                      if m in comparison_df.columns]
    
    if 'duration' in comparison_df.columns:
        comparison_df['Training Time (min)'] = comparison_df['duration'] / 60
        comparison_metrics.append('Training Time (min)')
    
    comparison_table = comparison_df[['model'] + comparison_metrics].set_index('model')
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=comparison_table.values,
        x=comparison_table.columns,
        y=comparison_table.index,
        colorscale='Viridis',
        text=comparison_table.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title="Model Comparison Heatmap",
        xaxis_title="Metrics",
        yaxis_title="Models",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def display_training_progress(df: pd.DataFrame, experiments_dir: Path):
    """Display training progress and curves."""
    st.header("Training Progress")
    
    # Select model
    model_options = df['model'].unique()
    selected_model = st.selectbox("Select model to view training progress", model_options)
    
    if not selected_model:
        return
    
    # Find log files
    model_data = df[df['model'] == selected_model].iloc[0]
    model_path = Path(model_data['model_path'])
    
    # Look for tensorboard logs or CSV metrics
    log_files = list(model_path.glob("**/metrics.csv")) + list(model_path.glob("**/training_log.csv"))
    
    if not log_files:
        st.info("No training logs found for this model")
        return
    
    # Load training data
    try:
        train_log = pd.read_csv(log_files[0])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Loss', 'Validation MAP@12', 'Learning Rate')
        )
        
        # Training loss
        if 'train_loss' in train_log.columns:
            fig.add_trace(
                go.Scatter(x=train_log.index, y=train_log['train_loss'], name='Train Loss'),
                row=1, col=1
            )
        
        # Validation loss
        if 'val_loss' in train_log.columns:
            fig.add_trace(
                go.Scatter(x=train_log.index, y=train_log['val_loss'], name='Val Loss'),
                row=1, col=2
            )
        
        # Validation MAP
        if 'val_map_at_k' in train_log.columns:
            fig.add_trace(
                go.Scatter(x=train_log.index, y=train_log['val_map_at_k'], name='Val MAP@12'),
                row=2, col=1
            )
        
        # Learning rate
        if 'lr' in train_log.columns:
            fig.add_trace(
                go.Scatter(x=train_log.index, y=train_log['lr'], name='Learning Rate'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title=f"Training Progress - {selected_model}")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading training logs: {e}")


def display_detailed_results(df: pd.DataFrame):
    """Display detailed results table."""
    st.header("Detailed Results")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_types = ['All'] + list(df['model_type'].unique())
        selected_type = st.selectbox("Filter by model type", model_types)
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ['MAP@12', 'model', 'timestamp'] + 
            [col for col in ['test_recall', 'test_precision', 'test_ndcg'] if col in df.columns],
            index=0
        )
    
    with col3:
        ascending = st.checkbox("Ascending order", value=False)
    
    # Filter data
    filtered_df = df.copy()
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['model_type'] == selected_type]
    
    # Sort
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
    
    # Display columns
    display_cols = ['model', 'model_type', 'MAP@12']
    for col in ['test_recall', 'test_precision', 'test_ndcg', 'duration']:
        if col in filtered_df.columns:
            display_cols.append(col)
    
    if 'timestamp' in filtered_df.columns:
        filtered_df['timestamp_str'] = filtered_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_cols.append('timestamp_str')
    
    # Round numeric columns
    numeric_cols = ['MAP@12', 'test_recall', 'test_precision', 'test_ndcg']
    for col in numeric_cols:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].round(4)
    
    if 'duration' in filtered_df.columns:
        filtered_df['duration'] = (filtered_df['duration'] / 60).round(1)
        filtered_df.rename(columns={'duration': 'Duration (min)'}, inplace=True)
        if 'Duration (min)' not in display_cols:
            display_cols[display_cols.index('duration')] = 'Duration (min)'
    
    # Display table
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        height=600
    )
    
    # Download button
    csv = filtered_df[display_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()