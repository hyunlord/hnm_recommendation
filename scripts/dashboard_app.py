"""Streamlit을 사용한 실험 모니터링을 위한 대화형 대시보드 앱."""
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

# 페이지 설정
st.set_page_config(
    page_title="H&M 추천 대시보드",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
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
    """캐싱과 함께 모든 실험 결과 로드."""
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
    
    # 결과 처리
    if len(df) > 0:
        if 'test_map' in df.columns:
            df['MAP@12'] = df['test_map']
        elif 'test_map_at_k' in df.columns:
            df['MAP@12'] = df['test_map_at_k']
        
        # 타임스탬프 추출
        df['timestamp'] = pd.to_datetime(
            df['experiment'].str.extract(r'(\d{8}_\d{6})')[0],
            format='%Y%m%d_%H%M%S',
            errors='coerce'
        )
        
        # 모델 타입
        df['model_type'] = df['model'].apply(
            lambda x: x.split('_')[0] if '_' in x else x
        )
    
    return df


def load_single_result(model_dir: Path) -> dict:
    """단일 모델 디렉토리에서 결과 로드."""
    result = {'model': model_dir.name}
    
    # 다른 결과 파일 형식 시도
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
    """메인 대시보드 애플리케이션."""
    st.title("🛍️ H&M 추천 시스템 대시보드")
    st.markdown("추천 모델 실험의 실시간 모니터링 및 분석")
    
    # 사이드바
    with st.sidebar:
        st.header("설정")
        
        # 실험 디렉토리
        experiments_dir = st.text_input(
            "실험 디렉토리",
            value="experiments",
            help="실험 디렉토리 경로"
        )
        experiments_path = Path(experiments_dir)
        
        # 새로고침 버튼
        if st.button("🔄 데이터 새로고침"):
            st.cache_data.clear()
            st.experimental_rerun()
        
        # 자동 새로고침
        auto_refresh = st.checkbox("자동 새로고침 (60초)", value=False)
        if auto_refresh:
            time.sleep(60)
            st.experimental_rerun()
    
    # 데이터 로드
    if not experiments_path.exists():
        st.error(f"실험 디렉토리를 찾을 수 없습니다: {experiments_path}")
        return
    
    with st.spinner("실험 결과 로드 중..."):
        results_df = load_experiment_results(experiments_path)
    
    if len(results_df) == 0:
        st.warning("실험 결과를 찾을 수 없습니다!")
        st.info("먼저 `python scripts/train.py` 또는 `python scripts/run_experiments.py`를 사용하여 실험을 실행하세요")
        return
    
    # 메인 콘텐츠
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 개요", 
        "📈 성능 분석", 
        "🔬 모델 비교",
        "📉 학습 진행 상황",
        "📋 상세 결과"
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
    """개요 메트릭 및 통계 표시."""
    st.header("개요")
    
    # 주요 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "전체 실험",
            len(df),
            delta=None
        )
    
    with col2:
        st.metric(
            "최고 MAP@12",
            f"{df['MAP@12'].max():.4f}" if 'MAP@12' in df.columns else "N/A",
            delta=None
        )
    
    with col3:
        st.metric(
            "고유 모델",
            df['model_type'].nunique() if 'model_type' in df.columns else 0,
            delta=None
        )
    
    with col4:
        latest_exp = df['timestamp'].max() if 'timestamp' in df.columns else None
        st.metric(
            "최신 실험",
            latest_exp.strftime("%Y-%m-%d %H:%M") if pd.notna(latest_exp) else "N/A",
            delta=None
        )
    
    # 최고 성능 모델
    st.subheader("🏆 상위 성능 모델")
    
    if 'MAP@12' in df.columns:
        top_models = df.nlargest(5, 'MAP@12')[['model', 'MAP@12', 'model_type']]
        
        # 막대 차트 생성
        fig = px.bar(
            top_models,
            x='MAP@12',
            y='model',
            orientation='h',
            color='model_type',
            title="MAP@12 기준 상위 5개 모델",
            labels={'MAP@12': 'MAP@12', 'model': '모델'},
            text='MAP@12'
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 최근 실험
    st.subheader("📅 최근 실험")
    if 'timestamp' in df.columns:
        recent = df.nlargest(10, 'timestamp')[['timestamp', 'model', 'MAP@12']]
        recent['timestamp'] = recent['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent, use_container_width=True)


def display_performance_analysis(df: pd.DataFrame):
    """성능 분석 시각화 표시."""
    st.header("성능 분석")
    
    if 'MAP@12' not in df.columns:
        st.warning("성능 메트릭을 찾을 수 없습니다")
        return
    
    # 모델 타입 비교
    col1, col2 = st.columns(2)
    
    with col1:
        # 모델 타입별 박스 플롯
        fig = px.box(
            df,
            x='model_type',
            y='MAP@12',
            title="모델 타입별 성능 분포",
            points="all"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 모델 타입별 평균 성능
        avg_perf = df.groupby('model_type')['MAP@12'].agg(['mean', 'std', 'count']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=avg_perf['model_type'],
            y=avg_perf['mean'],
            error_y=dict(type='data', array=avg_perf['std']),
            text=avg_perf['count'],
            texttemplate='n=%{text}',
            textposition='outside',
            name='평균 MAP@12'
        ))
        fig.update_layout(
            title="모델 타입별 평균 성능",
            xaxis_title="모델 타입",
            yaxis_title="MAP@12",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 시간에 따른 성능
    if 'timestamp' in df.columns and pd.notna(df['timestamp']).any():
        st.subheader("시간에 따른 성능")
        
        # 날짜 및 모델 타입별로 그룹화
        df['date'] = df['timestamp'].dt.date
        time_perf = df.groupby(['date', 'model_type'])['MAP@12'].mean().reset_index()
        
        fig = px.line(
            time_perf,
            x='date',
            y='MAP@12',
            color='model_type',
            title="시간에 따른 성능 추세",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 메트릭 상관관계
    metrics = ['MAP@12', 'test_recall', 'test_precision', 'test_ndcg']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) > 1:
        st.subheader("메트릭 상관관계")
        
        corr_matrix = df[available_metrics].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="상관관계"),
            x=available_metrics,
            y=available_metrics,
            color_continuous_scale='RdBu',
            aspect="auto",
            title="평가 메트릭 상관관계 행렬"
        )
        fig.update_traces(text=corr_matrix.values.round(3), texttemplate='%{text}')
        st.plotly_chart(fig, use_container_width=True)


def display_model_comparison(df: pd.DataFrame):
    """상세한 모델 비교 표시."""
    st.header("모델 비교")
    
    # 모델 선택
    available_models = df['model'].unique()
    selected_models = st.multiselect(
        "비교할 모델 선택",
        available_models,
        default=list(df.nlargest(5, 'MAP@12')['model']) if 'MAP@12' in df.columns else []
    )
    
    if not selected_models:
        st.info("비교할 모델을 선택하세요")
        return
    
    # 데이터 필터링
    comparison_df = df[df['model'].isin(selected_models)]
    
    # 레이더 차트
    metrics = ['MAP@12', 'test_recall', 'test_precision', 'test_ndcg']
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    if len(available_metrics) >= 3:
        st.subheader("다중 메트릭 비교")
        
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
            title="모델 성능 레이더 차트"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 나란히 비교
    st.subheader("나란히 비교")
    
    comparison_metrics = ['MAP@12'] + [m for m in ['test_recall', 'test_precision', 'test_ndcg'] 
                                      if m in comparison_df.columns]
    
    if 'duration' in comparison_df.columns:
        comparison_df['학습 시간 (분)'] = comparison_df['duration'] / 60
        comparison_metrics.append('학습 시간 (분)')
    
    comparison_table = comparison_df[['model'] + comparison_metrics].set_index('model')
    
    # 히트맵 생성
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
        title="모델 비교 히트맵",
        xaxis_title="메트릭",
        yaxis_title="모델",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def display_training_progress(df: pd.DataFrame, experiments_dir: Path):
    """학습 진행 상황 및 곡선 표시."""
    st.header("학습 진행 상황")
    
    # 모델 선택
    model_options = df['model'].unique()
    selected_model = st.selectbox("학습 진행 상황을 볼 모델 선택", model_options)
    
    if not selected_model:
        return
    
    # 로그 파일 찾기
    model_data = df[df['model'] == selected_model].iloc[0]
    model_path = Path(model_data['model_path'])
    
    # 텐서보드 로그 또는 CSV 메트릭 찾기
    log_files = list(model_path.glob("**/metrics.csv")) + list(model_path.glob("**/training_log.csv"))
    
    if not log_files:
        st.info("이 모델에 대한 학습 로그를 찾을 수 없습니다")
        return
    
    # 학습 데이터 로드
    try:
        train_log = pd.read_csv(log_files[0])
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('학습 손실', '검증 손실', '검증 MAP@12', '학습률')
        )
        
        # 학습 손실
        if 'train_loss' in train_log.columns:
            fig.add_trace(
                go.Scatter(x=train_log.index, y=train_log['train_loss'], name='학습 손실'),
                row=1, col=1
            )
        
        # 검증 손실
        if 'val_loss' in train_log.columns:
            fig.add_trace(
                go.Scatter(x=train_log.index, y=train_log['val_loss'], name='검증 손실'),
                row=1, col=2
            )
        
        # 검증 MAP
        if 'val_map_at_k' in train_log.columns:
            fig.add_trace(
                go.Scatter(x=train_log.index, y=train_log['val_map_at_k'], name='검증 MAP@12'),
                row=2, col=1
            )
        
        # 학습률
        if 'lr' in train_log.columns:
            fig.add_trace(
                go.Scatter(x=train_log.index, y=train_log['lr'], name='학습률'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title=f"학습 진행 상황 - {selected_model}")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"학습 로그 로드 오류: {e}")


def display_detailed_results(df: pd.DataFrame):
    """상세 결과 테이블 표시."""
    st.header("상세 결과")
    
    # 필터
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_types = ['전체'] + list(df['model_type'].unique())
        selected_type = st.selectbox("모델 타입별 필터", model_types)
    
    with col2:
        sort_by = st.selectbox(
            "정렬 기준",
            ['MAP@12', 'model', 'timestamp'] + 
            [col for col in ['test_recall', 'test_precision', 'test_ndcg'] if col in df.columns],
            index=0
        )
    
    with col3:
        ascending = st.checkbox("오름차순 정렬", value=False)
    
    # 데이터 필터링
    filtered_df = df.copy()
    if selected_type != '전체':
        filtered_df = filtered_df[filtered_df['model_type'] == selected_type]
    
    # 정렬
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
    
    # 표시 커럼
    display_cols = ['model', 'model_type', 'MAP@12']
    for col in ['test_recall', 'test_precision', 'test_ndcg', 'duration']:
        if col in filtered_df.columns:
            display_cols.append(col)
    
    if 'timestamp' in filtered_df.columns:
        filtered_df['timestamp_str'] = filtered_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_cols.append('timestamp_str')
    
    # 숫자 커럼 반올림
    numeric_cols = ['MAP@12', 'test_recall', 'test_precision', 'test_ndcg']
    for col in numeric_cols:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].round(4)
    
    if 'duration' in filtered_df.columns:
        filtered_df['duration'] = (filtered_df['duration'] / 60).round(1)
        filtered_df.rename(columns={'duration': '소요 시간 (분)'}, inplace=True)
        if '소요 시간 (분)' not in display_cols:
            display_cols[display_cols.index('duration')] = '소요 시간 (분)'
    
    # 테이블 표시
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        height=600
    )
    
    # 다운로드 버튼
    csv = filtered_df[display_cols].to_csv(index=False)
    st.download_button(
        label="📥 결과를 CSV로 다운로드",
        data=csv,
        file_name=f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()