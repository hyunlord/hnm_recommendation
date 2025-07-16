"""H&M 추천 모델 학습 스크립트."""
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, MLFlowLogger

from src.data import HMDataModule, ImprovedHMDataModule
from src.models import (
    PopularityBaseline,
    MatrixFactorization,
    NeuralCF,
    WideDeep,
    LightGCN,
)
from src.utils import set_seed


log = logging.getLogger(__name__)


def get_logger(cfg: DictConfig) -> Optional[pl.loggers.Logger]:
    """설정에 기반하여 로거 생성.
    
    Args:
        cfg: 설정 객체
        
    Returns:
        로거 인스턴스 또는 None
    """
    if not cfg.logging.enabled:
        return None
    
    logger_type = cfg.logging.logger
    
    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=cfg.paths.log_dir,
            name=cfg.model.name,
            version=cfg.run_name,
        )
    elif logger_type == "wandb":
        return WandbLogger(
            project=cfg.project.name,
            name=cfg.run_name,
            save_dir=cfg.paths.log_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    elif logger_type == "mlflow":
        return MLFlowLogger(
            experiment_name=cfg.project.name,
            run_name=cfg.run_name,
            save_dir=cfg.paths.log_dir,
        )
    else:
        return None


def get_callbacks(cfg: DictConfig) -> list:
    """설정에 기반하여 콜백 생성.
    
    Args:
        cfg: 설정 객체
        
    Returns:
        콜백 리스트
    """
    callbacks = []
    
    # 모델 체크포인트
    callbacks.append(
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            filename=f"{cfg.model.name}-{{epoch:02d}}-{{val_map_at_k:.4f}}",
            monitor="val_map_at_k",
            mode="max",
            save_top_k=cfg.training.save_top_k,
            save_last=True,
            verbose=True,
        )
    )
    
    # 조기 종료
    if cfg.training.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_map_at_k",
                mode="max",
                patience=cfg.training.patience,
                verbose=True,
            )
        )
    
    # 학습률 모니터
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    
    # 진행률 표시줄
    callbacks.append(RichProgressBar())
    
    return callbacks


def instantiate_model(cfg: DictConfig, data_module: HMDataModule) -> pl.LightningModule:
    """설정에 기반하여 모델 인스턴스 생성.
    
    Args:
        cfg: 설정 객체
        data_module: 데이터 모듈 인스턴스
        
    Returns:
        모델 인스턴스
    """
    # 데이터 정보로 모델 설정 업데이트
    model_cfg = cfg.model.copy()
    model_cfg.num_users = data_module.num_users
    model_cfg.num_items = data_module.num_items
    
    # Wide & Deep 모델을 위한 특징 차원 추가
    if cfg.model.name == "wide_deep":
        model_cfg.num_user_features = data_module.num_user_features
        model_cfg.num_item_features = data_module.num_item_features
    
    # name 필드는 모델 파라미터가 아니므로 제거
    model_cfg.pop("name", None)
    
    # Hydra의 instantiate를 사용하거나 직접 모델 인스턴스 생성
    model_class = {
        "popularity_baseline": PopularityBaseline,
        "matrix_factorization": MatrixFactorization,
        "neural_cf": NeuralCF,
        "wide_deep": WideDeep,
        "lightgcn": LightGCN,
    }.get(cfg.model.name)
    
    if model_class is None:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    
    return model_class(**model_cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> Dict[str, float]:
    """추천 모델 학습.
    
    Args:
        cfg: 설정 객체
        
    Returns:
        최종 메트릭 딕셔너리
    """
    # 재현성을 위한 시드 설정
    set_seed(cfg.project.seed)
    
    # 설정 로깅
    log.info("다음 설정으로 학습 시작:")
    log.info(OmegaConf.to_yaml(cfg))
    
    # 데이터 모듈 생성
    log.info("데이터 모듈 생성 중...")
    
    # 지정된 경우 개선된 데이터 모듈 사용
    use_improved = getattr(cfg.data, 'use_improved_datamodule', True)
    
    if use_improved:
        data_module = ImprovedHMDataModule(
            data_dir=cfg.paths.data_dir,
            processed_dir=f"{cfg.paths.data_dir}/processed",
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            negative_sampling_ratio=cfg.data.negative_sampling_ratio,
            min_user_interactions=cfg.data.min_user_interactions,
            min_item_interactions=cfg.data.min_item_interactions,
            train_weeks=cfg.data.train_weeks,
            val_weeks=cfg.data.val_weeks,
            test_weeks=cfg.data.test_weeks,
            sample_fraction=getattr(cfg.data, 'sample_fraction', 1.0),
            use_features=getattr(cfg.data, 'use_features', False),
            dataset_type=getattr(cfg.data, 'dataset_type', 'standard'),
            sampling_strategy=getattr(cfg.data, 'sampling_strategy', 'uniform'),
            cache_negatives=getattr(cfg.data, 'cache_negatives', True),
            temporal_window_days=getattr(cfg.data, 'temporal_window_days', 7),
            augment_data=getattr(cfg.data, 'augment_data', False),
            normalize_features=getattr(cfg.data, 'normalize_features', True),
        )
    else:
        data_module = HMDataModule(
            data_dir=cfg.paths.data_dir,
            processed_dir=f"{cfg.paths.data_dir}/processed",
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            negative_sampling_ratio=cfg.data.negative_sampling_ratio,
            min_user_interactions=cfg.data.min_user_interactions,
            min_item_interactions=cfg.data.min_item_interactions,
            train_weeks=cfg.data.train_weeks,
            val_weeks=cfg.data.val_weeks,
            test_weeks=cfg.data.test_weeks,
            sample_fraction=getattr(cfg.data, 'sample_fraction', 1.0),
            use_features=getattr(cfg.data, 'use_features', False),
        )
    
    # 데이터 설정 (필요시 로드 및 전처리)
    data_module.setup()
    
    # 모델 생성
    log.info(f"모델 생성 중: {cfg.model.name}")
    model = instantiate_model(cfg, data_module)
    
    # LightGCN을 위한 특별 처리 - 그래프 구조 설정
    if cfg.model.name == "lightgcn":
        log.info("LightGCN을 위한 그래프 구축 중...")
        edge_index, edge_weight = data_module.get_graph()
        model.set_graph(edge_index, edge_weight)
    
    # 로거 생성
    logger = get_logger(cfg)
    
    # 콜백 생성
    callbacks = get_callbacks(cfg)
    
    # 트레이너 생성
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.project.device,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        deterministic=True,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        num_sanity_val_steps=2,
    )
    
    # 모델 학습
    log.info("학습 시작...")
    trainer.fit(model, data_module)
    
    # 모델 테스트
    log.info("모델 테스트 중...")
    test_results = trainer.test(model, data_module, ckpt_path="best")
    
    # 최종 결과 저장
    results = {
        "best_val_map": trainer.checkpoint_callback.best_model_score.item(),
        "test_map": test_results[0]["test_map_at_k"],
        "test_recall": test_results[0]["test_recall_at_k"],
        "test_precision": test_results[0]["test_precision_at_k"],
        "test_ndcg": test_results[0]["test_ndcg_at_k"],
    }
    
    # 파일에 결과 저장
    results_path = Path(cfg.paths.results_dir) / f"{cfg.run_name}_results.yaml"
    OmegaConf.save(results, results_path)
    
    log.info("학습 완료!")
    log.info(f"결과: {results}")
    
    return results


if __name__ == "__main__":
    train()