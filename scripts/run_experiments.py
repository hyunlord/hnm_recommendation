"""포괄적인 모델 비교 실험을 실행하는 스크립트."""
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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """모델 비교 실험을 관리하고 실행."""
    
    def __init__(
        self, 
        base_dir: str = "experiments",
        sample_fraction: float = 0.1,
        quick_test: bool = False
    ):
        """실험 러너 초기화.
        
        Args:
            base_dir: 실험 결과를 위한 기본 디렉토리
            sample_fraction: 사용할 데이터 비율 (빠른 실험을 위해)
            quick_test: True인 경우, 적은 epoch로 빠른 테스트 실행
        """
        self.base_dir = Path(base_dir)
        self.sample_fraction = sample_fraction
        self.quick_test = quick_test
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / f"comparison_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 정의
        self.experiments = self._define_experiments()
        
    def _define_experiments(self) -> List[Dict]:
        """실행할 모든 실험 정의."""
        experiments = []
        
        # 기본 설정
        base_config = {
            "data.sample_fraction": self.sample_fraction,
            "training.num_workers": 4,
            "logging.enabled": True,
        }
        
        if self.quick_test:
            base_config["training.epochs"] = 5
            base_config["training.val_check_interval"] = 1.0
        
        # 1. 인기도 베이스라인
        experiments.append({
            "name": "popularity_baseline",
            "model": "popularity_baseline",
            "config": {**base_config}
        })
        
        # 2. 행렬 분해
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
        
        # 3. 신경망 협업 필터링
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
        
        # 6. 다른 샘플링 전략을 사용한 고급 실험
        if not self.quick_test:
            # 인기도 샘플링을 사용한 Neural CF
            experiments.append({
                "name": "neural_cf_popularity",
                "model": "neural_cf",
                "config": {
                    **base_config,
                    "data.sampling_strategy": "popularity",
                    "data.negative_sampling_ratio": 4,
                }
            })
            
            # 하드 네거티브 샘플링을 사용한 Neural CF
            experiments.append({
                "name": "neural_cf_hard",
                "model": "neural_cf",
                "config": {
                    **base_config,
                    "data.sampling_strategy": "hard",
                    "data.negative_sampling_ratio": 4,
                }
            })
            
            # 특징 없는 Wide & Deep
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
        """단일 실험 실행.
        
        Args:
            experiment: 실험 설정
            
        Returns:
            결과 딕셔너리
        """
        name = experiment["name"]
        logger.info(f"실험 실행 중: {name}")
        
        # 실험 디렉토리 생성
        exp_dir = self.experiment_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        # 명령어 구성
        cmd = ["python", "scripts/train.py"]
        cmd.append(f"model={experiment['model']}")
        
        # 설정 오버라이드 추가
        for key, value in experiment["config"].items():
            cmd.append(f"{key}={value}")
        
        # 출력 디렉토리 추가
        cmd.append(f"paths.output_dir={exp_dir}")
        cmd.append(f"run_name={name}")
        
        # 실험 실행
        logger.info(f"명령어: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            duration = time.time() - start_time
            
            # 출력에서 결과 파싱
            results = self._parse_results(result.stdout, exp_dir)
            results["duration"] = duration
            results["status"] = "success"
            results["name"] = name
            results["model"] = experiment["model"]
            
            logger.info(f"실험 {name} 성공적으로 완료")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"실험 {name} 실패: {e}")
            logger.error(f"오류 출력: {e.stderr}")
            results = {
                "name": name,
                "model": experiment["model"],
                "status": "failed",
                "error": str(e),
                "duration": time.time() - start_time
            }
        
        # 개별 실험 결과 저장
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _parse_results(self, output: str, exp_dir: Path) -> Dict:
        """학습 출력에서 결과 파싱.
        
        Args:
            output: 학습 스크립트 출력
            exp_dir: 실험 디렉토리
            
        Returns:
            파싱된 결과
        """
        results = {}
        
        # 저장된 YAML 파일에서 결과 로드 시도
        results_files = list(exp_dir.glob("*_results.yaml"))
        if results_files:
            with open(results_files[0], "r") as f:
                saved_results = yaml.safe_load(f)
                results.update(saved_results)
        
        # 백업으로 출력에서 파싱
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
        """정의된 모든 실험 실행.
        
        Returns:
            모든 결과가 있는 DataFrame
        """
        logger.info(f"{len(self.experiments)}개의 실험 시작")
        logger.info(f"결과는 다음 경로에 저장됨: {self.experiment_dir}")
        
        all_results = []
        
        for i, experiment in enumerate(self.experiments):
            logger.info(f"\n{'='*60}")
            logger.info(f"실험 {i+1}/{len(self.experiments)}: {experiment['name']}")
            logger.info(f"{'='*60}")
            
            results = self.run_experiment(experiment)
            all_results.append(results)
            
            # 중간 결과 저장
            df = pd.DataFrame(all_results)
            df.to_csv(self.experiment_dir / "results_intermediate.csv", index=False)
        
        # 최종 결과 DataFrame 생성
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.experiment_dir / "results_final.csv", index=False)
        
        logger.info("\n모든 실험 완료!")
        return results_df
    
    def create_visualizations(self, results_df: pd.DataFrame):
        """실험 결과를 위한 시각화 플롯 생성.
        
        Args:
            results_df: 실험 결과가 있는 DataFrame
        """
        logger.info("시각화 생성 중...")
        
        # 성공한 실험 필터링
        success_df = results_df[results_df["status"] == "success"].copy()
        
        if len(success_df) == 0:
            logger.warning("시각화할 성공한 실험이 없음")
            return
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. 전체 성능 비교
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ["test_map_at_k", "test_recall_at_k", "test_precision", "test_ndcg_at_k"]
        titles = ["MAP@12", "Recall@12", "Precision@12", "NDCG@12"]
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            if metric in success_df.columns:
                data = success_df[["name", metric]].dropna()
                if len(data) > 0:
                    ax.bar(data["name"], data[metric])
                    ax.set_title(f"{title} 비교", fontsize=14)
                    ax.set_xlabel("모델", fontsize=12)
                    ax.set_ylabel(title, fontsize=12)
                    ax.tick_params(axis='x', rotation=45)
                    
                    # 값 레이블 추가
                    for i, v in enumerate(data[metric]):
                        ax.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "performance_comparison.png", dpi=300)
        plt.close()
        
        # 2. 학습 시간 비교
        if "duration" in success_df.columns:
            plt.figure(figsize=(12, 6))
            success_df["duration_min"] = success_df["duration"] / 60
            ax = sns.barplot(data=success_df, x="name", y="duration_min")
            plt.title("학습 시간 비교", fontsize=14)
            plt.xlabel("모델", fontsize=12)
            plt.ylabel("학습 시간 (분)", fontsize=12)
            plt.xticks(rotation=45)
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}',
                           (p.get_x() + p.get_width()/2., p.get_height()),
                           ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.experiment_dir / "training_time_comparison.png", dpi=300)
            plt.close()
        
        # 3. 요약 테이블 생성
        if "test_map_at_k" in success_df.columns:
            summary_metrics = ["test_map_at_k", "test_recall_at_k", "test_ndcg_at_k", "duration_min"]
            summary_df = success_df[["name", "model"] + 
                                  [m for m in summary_metrics if m in success_df.columns]]
            
            # 숫자값 반올림
            for col in summary_metrics:
                if col in summary_df.columns:
                    if col == "duration_min":
                        summary_df[col] = summary_df[col].round(1)
                    else:
                        summary_df[col] = summary_df[col].round(4)
            
            # MAP@12로 정렬
            if "test_map_at_k" in summary_df.columns:
                summary_df = summary_df.sort_values("test_map_at_k", ascending=False)
            
            # 마크다운 테이블로 저장
            with open(self.experiment_dir / "summary_table.md", "w") as f:
                f.write("# 모델 성능 요약\n\n")
                f.write(summary_df.to_markdown(index=False))
        
        logger.info(f"시각화 저장 경로: {self.experiment_dir}")
    
    def create_report(self, results_df: pd.DataFrame):
        """포괄적인 실험 보고서 생성.
        
        Args:
            results_df: 실험 결과가 있는 DataFrame
        """
        logger.info("실험 보고서 생성 중...")
        
        report_path = self.experiment_dir / "experiment_report.md"
        
        with open(report_path, "w") as f:
            f.write(f"# H&M 추천 모델 비교 보고서\n\n")
            f.write(f"**날짜**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**샘플 비율**: {self.sample_fraction}\n")
            f.write(f"**빠른 테스트**: {self.quick_test}\n\n")
            
            # 요약 통계
            success_df = results_df[results_df["status"] == "success"]
            f.write(f"## 요약\n\n")
            f.write(f"- 전체 실험: {len(results_df)}\n")
            f.write(f"- 성공: {len(success_df)}\n")
            f.write(f"- 실패: {len(results_df) - len(success_df)}\n\n")
            
            # 최고 성능 모델
            if len(success_df) > 0 and "test_map_at_k" in success_df.columns:
                best_model = success_df.loc[success_df["test_map_at_k"].idxmax()]
                f.write(f"## 최고 성능 모델\n\n")
                f.write(f"**{best_model['name']}**이(가) 최고 MAP@12 달성: {best_model['test_map_at_k']:.4f}\n\n")
            
            # 성능 테이블
            if len(success_df) > 0:
                f.write(f"## 성능 비교\n\n")
                summary_cols = ["name", "test_map_at_k", "test_recall_at_k", "test_ndcg_at_k"]
                summary_cols = [col for col in summary_cols if col in success_df.columns]
                
                if summary_cols:
                    summary_df = success_df[summary_cols].round(4)
                    if "test_map_at_k" in summary_df.columns:
                        summary_df = summary_df.sort_values("test_map_at_k", ascending=False)
                    f.write(summary_df.to_markdown(index=False))
                    f.write("\n\n")
            
            # 실패한 실험
            failed_df = results_df[results_df["status"] == "failed"]
            if len(failed_df) > 0:
                f.write(f"## 실패한 실험\n\n")
                for _, row in failed_df.iterrows():
                    f.write(f"- **{row['name']}**: {row.get('error', '알 수 없는 오류')}\n")
                f.write("\n")
            
            # 권장사항
            f.write(f"## 권장사항\n\n")
            if len(success_df) > 0 and "test_map_at_k" in success_df.columns:
                top_models = success_df.nlargest(3, "test_map_at_k")
                f.write("실험 결과에 따른 상위 3개 모델:\n\n")
                for i, (_, model) in enumerate(top_models.iterrows(), 1):
                    f.write(f"{i}. **{model['name']}**: MAP@12 = {model['test_map_at_k']:.4f}\n")
            
        logger.info(f"보고서 저장 경로: {report_path}")


def main():
    """실험을 실행하는 메인 함수."""
    import argparse
    
    parser = argparse.ArgumentParser(description="H&M 추천 모델 실험 실행")
    parser.add_argument("--sample-fraction", type=float, default=0.1,
                       help="사용할 데이터 비율 (기본값: 0.1)")
    parser.add_argument("--quick-test", action="store_true",
                       help="적은 epoch로 빠른 테스트 실행")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                       help="실험을 위한 기본 디렉토리 (기본값: experiments)")
    
    args = parser.parse_args()
    
    # 실험 러너 생성
    runner = ExperimentRunner(
        base_dir=args.experiments_dir,
        sample_fraction=args.sample_fraction,
        quick_test=args.quick_test
    )
    
    # 모든 실험 실행
    results_df = runner.run_all_experiments()
    
    # 시각화 및 보고서 생성
    runner.create_visualizations(results_df)
    runner.create_report(results_df)
    
    logger.info("\n실험 비교 완료!")
    logger.info(f"결과 저장 경로: {runner.experiment_dir}")


if __name__ == "__main__":
    main()