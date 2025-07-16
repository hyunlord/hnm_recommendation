"""H&M 추천 모델 서빙을 위한 FastAPI 서버."""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime
import pickle

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.models import (
    PopularityBaseline,
    MatrixFactorization,
    NeuralCF,
    WideDeep,
    LightGCN,
)
from src.data import ImprovedHMDataModule
from src.utils import set_seed

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="H&M 추천 API",
    description="H&M 패션 추천을 위한 API",
    version="1.0.0"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청/응답 모델
class RecommendationRequest(BaseModel):
    """추천 요청 모델."""
    user_id: Union[int, str] = Field(..., description="사용자 ID (고객 ID 또는 인덱스)")
    num_items: int = Field(12, ge=1, le=100, description="추천할 아이템 수")
    filter_purchased: bool = Field(True, description="이미 구매한 아이템 필터링")
    include_scores: bool = Field(False, description="추천 점수 포함")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if isinstance(v, str):
            # H&M 고객 ID 형식 처리
            if not v.replace('.', '').isdigit():
                raise ValueError('잘못된 사용자 ID 형식')
        return v


class BatchRecommendationRequest(BaseModel):
    """배치 추천 요청 모델."""
    user_ids: List[Union[int, str]] = Field(..., description="사용자 ID 리스트")
    num_items: int = Field(12, ge=1, le=100, description="사용자당 아이템 수")
    filter_purchased: bool = Field(True, description="이미 구매한 아이템 필터링")
    include_scores: bool = Field(False, description="추천 점수 포함")


class ItemInfo(BaseModel):
    """아이템 정보 모델."""
    article_id: str
    product_name: Optional[str] = None
    product_type_name: Optional[str] = None
    product_group_name: Optional[str] = None
    colour_group_name: Optional[str] = None
    department_name: Optional[str] = None
    score: Optional[float] = None


class RecommendationResponse(BaseModel):
    """추천 응답 모델."""
    user_id: Union[int, str]
    recommendations: List[ItemInfo]
    model_name: str
    generated_at: str


class ModelInfo(BaseModel):
    """모델 정보."""
    name: str
    type: str
    checkpoint_path: Optional[str]
    metrics: Optional[Dict[str, float]]
    loaded: bool


class HealthResponse(BaseModel):
    """상태 검사 응답."""
    status: str
    models_loaded: int
    available_models: List[str]
    data_loaded: bool


class ModelServer:
    """추천 처리를 위한 모델 서버."""
    
    def __init__(
        self,
        data_dir: str = "data",
        checkpoint_dir: str = "experiments/checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """모델 서버 초기화.
        
        Args:
            data_dir: 데이터 디렉토리 경로
            checkpoint_dir: 모델 체크포인트가 있는 디렉토리
            device: 추론에 사용할 디바이스
        """
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        
        self.models = {}
        self.model_info = {}
        self.data_module = None
        self.encoders = None
        self.article_info = None
        self.user_history = {}
        
        # 초기화
        self._load_data()
        self._load_models()
    
    def _load_data(self):
        """데이터 및 전처리된 정보 로드."""
        logger.info("데이터 로드 중...")
        
        # 데이터 모듈 로드
        self.data_module = ImprovedHMDataModule(
            data_dir=str(self.data_dir),
            batch_size=1024,
            num_workers=0,  # 서빙용
        )
        self.data_module.setup()
        
        # 인코더 로드
        encoder_path = self.data_dir / "processed" / "encoders.pkl"
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.encoders = pickle.load(f)
        
        # 아티클 정보 로드
        articles_path = self.data_dir / "processed" / "articles.parquet"
        if articles_path.exists():
            self.article_info = pd.read_parquet(articles_path)
        
        # 사용자 구매 이력 로드
        self._load_user_history()
        
        logger.info(f"데이터 로드 완료: {self.data_module.num_users}명의 사용자, {self.data_module.num_items}개의 아이템")
    
    def _load_user_history(self):
        """필터링을 위한 사용자 구매 이력 로드."""
        train_path = self.data_dir / "processed" / "train.parquet"
        if train_path.exists():
            train_df = pd.read_parquet(train_path)
            self.user_history = train_df.groupby('customer_idx')['article_idx'].apply(set).to_dict()
    
    def _load_models(self):
        """체크포인트에서 학습된 모델 로드."""
        logger.info("모델 로드 중...")
        
        # 사용 가능한 체크포인트 찾기
        if self.checkpoint_dir.exists():
            for ckpt_path in self.checkpoint_dir.glob("**/*.ckpt"):
                try:
                    model_name = ckpt_path.parent.name
                    logger.info(f"{model_name}을 {ckpt_path}에서 로드 중")
                    
                    # 체크포인트 로드
                    checkpoint = torch.load(ckpt_path, map_location=self.device)
                    
                    # 모델 인스턴스 생성
                    model = self._create_model_from_checkpoint(model_name, checkpoint)
                    
                    if model is not None:
                        self.models[model_name] = model
                        self.model_info[model_name] = ModelInfo(
                            name=model_name,
                            type=type(model).__name__,
                            checkpoint_path=str(ckpt_path),
                            metrics=checkpoint.get('metrics', {}),
                            loaded=True
                        )
                        logger.info(f"{model_name} 로드 성공")
                
                except Exception as e:
                    logger.error(f"{model_name} 로드 실패: {e}")
        
        # 폴백으로 인기도 베이스라인 로드
        if "popularity_baseline" not in self.models:
            self._load_popularity_baseline()
        
        logger.info(f"{len(self.models)}개의 모델 로드 완료")
    
    def _create_model_from_checkpoint(self, model_name: str, checkpoint: Dict) -> Optional[torch.nn.Module]:
        """체크포인트에서 모델 인스턴스 생성.
        
        Args:
            model_name: 모델 이름
            checkpoint: 체크포인트 딕셔너리
            
        Returns:
            모델 인스턴스 또는 None
        """
        try:
            state_dict = checkpoint['state_dict']
            hparams = checkpoint.get('hyper_parameters', {})
            
            # 데이터 차원으로 업데이트
            hparams['num_users'] = self.data_module.num_users
            hparams['num_items'] = self.data_module.num_items
            
            # 타입에 따른 모델 생성
            if 'matrix_factorization' in model_name:
                model = MatrixFactorization(**hparams)
            elif 'neural_cf' in model_name:
                model = NeuralCF(**hparams)
            elif 'wide_deep' in model_name:
                model = WideDeep(**hparams)
            elif 'lightgcn' in model_name:
                model = LightGCN(**hparams)
                # LightGCN을 위한 그래프 설정
                edge_index, edge_weight = self.data_module.get_graph()
                model.set_graph(edge_index.to(self.device), edge_weight)
            else:
                return None
            
            # state dict 로드
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"체크포인트에서 모델 생성 오류: {e}")
            return None
    
    def _load_popularity_baseline(self):
        """인기도 베이스라인 모델 로드."""
        logger.info("인기도 베이스라인 로드 중...")
        
        model = PopularityBaseline(
            num_items=self.data_module.num_items,
            top_k=100
        )
        
        # 인기 아이템 가져오기
        popular_items = self.data_module.get_popular_items(k=1000)
        model.set_popular_items(popular_items)
        
        self.models["popularity_baseline"] = model
        self.model_info["popularity_baseline"] = ModelInfo(
            name="popularity_baseline",
            type="PopularityBaseline",
            checkpoint_path=None,
            metrics=None,
            loaded=True
        )
    
    def get_user_idx(self, user_id: Union[int, str]) -> Optional[int]:
        """사용자 ID를 인덱스로 변환.
        
        Args:
            user_id: 사용자 ID (고객 ID 또는 인덱스)
            
        Returns:
            사용자 인덱스 또는 None
        """
        if isinstance(user_id, int):
            return user_id if user_id < self.data_module.num_users else None
        
        # 고객 ID를 인덱스로 변환
        if self.encoders and 'customer_id' in self.encoders:
            try:
                return self.encoders['customer_id'].transform([user_id])[0]
            except:
                return None
        
        return None
    
    def get_recommendations(
        self,
        user_id: Union[int, str],
        model_name: str = "best",
        num_items: int = 12,
        filter_purchased: bool = True,
        include_scores: bool = False
    ) -> Dict:
        """사용자를 위한 추천 생성.
        
        Args:
            user_id: 사용자 ID
            model_name: 사용할 모델 (최고 성능 모델의 경우 "best")
            num_items: 추천할 아이템 수
            filter_purchased: 구매한 아이템 필터링
            include_scores: 추천 점수 포함
            
        Returns:
            추천 결과
        """
        # 사용자 인덱스 가져오기
        user_idx = self.get_user_idx(user_id)
        if user_idx is None:
            raise ValueError(f"사용자 {user_id}를 찾을 수 없음")
        
        # 모델 선택
        if model_name == "best":
            # 메트릭에 기반한 최고 성능 모델 사용
            best_model_name = self._get_best_model()
            model = self.models[best_model_name]
            model_name = best_model_name
        elif model_name in self.models:
            model = self.models[model_name]
        else:
            raise ValueError(f"모델 {model_name}을 사용할 수 없음")
        
        # 추천 생성
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            
            # 점수 계산
            if hasattr(model, 'predict_all_items'):
                scores = model.predict_all_items(user_tensor)
            else:
                scores = model(user_tensor)
            
            # 구매한 아이템 필터링
            if filter_purchased and user_idx in self.user_history:
                purchased_items = list(self.user_history[user_idx])
                scores[0, purchased_items] = float('-inf')
            
            # 상위 아이템 가져오기
            top_scores, top_indices = torch.topk(scores[0], num_items)
            top_items = top_indices.cpu().numpy()
            top_scores = top_scores.cpu().numpy()
        
        # 결과 포맷팅
        recommendations = []
        for item_idx, score in zip(top_items, top_scores):
            item_info = self._get_item_info(item_idx)
            if include_scores:
                item_info.score = float(score)
            recommendations.append(item_info)
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "model_name": model_name,
            "generated_at": datetime.now().isoformat()
        }
    
    def get_batch_recommendations(
        self,
        user_ids: List[Union[int, str]],
        model_name: str = "best",
        num_items: int = 12,
        filter_purchased: bool = True,
        include_scores: bool = False
    ) -> List[Dict]:
        """여러 사용자를 위한 추천 생성.
        
        Args:
            user_ids: 사용자 ID 리스트
            model_name: 사용할 모델
            num_items: 사용자당 아이템 수
            filter_purchased: 구매한 아이템 필터링
            include_scores: 추천 점수 포함
            
        Returns:
            추천 결과 리스트
        """
        results = []
        
        for user_id in user_ids:
            try:
                result = self.get_recommendations(
                    user_id=user_id,
                    model_name=model_name,
                    num_items=num_items,
                    filter_purchased=filter_purchased,
                    include_scores=include_scores
                )
                results.append(result)
            except Exception as e:
                logger.error(f"사용자 {user_id}에 대한 추천 생성 오류: {e}")
                results.append({
                    "user_id": user_id,
                    "error": str(e)
                })
        
        return results
    
    def _get_best_model(self) -> str:
        """메트릭에 기반한 최고 성능 모델 가져오기.
        
        Returns:
            최고 모델 이름
        """
        best_model = "popularity_baseline"
        best_score = 0
        
        for model_name, info in self.model_info.items():
            if info.metrics and 'test_map' in info.metrics:
                if info.metrics['test_map'] > best_score:
                    best_score = info.metrics['test_map']
                    best_model = model_name
        
        return best_model
    
    def _get_item_info(self, item_idx: int) -> ItemInfo:
        """아이템 정보 가져오기.
        
        Args:
            item_idx: 아이템 인덱스
            
        Returns:
            아이템 정보
        """
        if self.article_info is not None and item_idx < len(self.article_info):
            row = self.article_info.iloc[item_idx]
            
            # 아티클 ID 가져오기
            if 'article_id' in row:
                article_id = str(row['article_id'])
            else:
                # 인덱스에서 디코드
                if self.encoders and 'article_id' in self.encoders:
                    article_id = self.encoders['article_id'].inverse_transform([item_idx])[0]
                else:
                    article_id = str(item_idx)
            
            return ItemInfo(
                article_id=article_id,
                product_name=row.get('prod_name', None),
                product_type_name=row.get('product_type_name', None),
                product_group_name=row.get('product_group_name', None),
                colour_group_name=row.get('colour_group_name', None),
                department_name=row.get('department_name', None)
            )
        else:
            return ItemInfo(article_id=str(item_idx))


# 모델 서버 초기화
model_server = None


@app.on_event("startup")
async def startup_event():
    """시작 시 모델 서버 초기화."""
    global model_server
    logger.info("모델 서버 시작 중...")
    
    # 환경 변수 또는 기본값으로 초기화
    data_dir = os.getenv("DATA_DIR", "data")
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "experiments/checkpoints")
    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    model_server = ModelServer(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    
    logger.info("모델 서버 준비 완료!")


@app.get("/", response_model=Dict)
async def root():
    """루트 엔드포인트."""
    return {
        "message": "H&M 추천 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """상태 검사 엔드포인트."""
    if model_server is None:
        raise HTTPException(status_code=503, detail="모델 서버가 초기화되지 않음")
    
    return HealthResponse(
        status="healthy",
        models_loaded=len(model_server.models),
        available_models=list(model_server.models.keys()),
        data_loaded=model_server.data_module is not None
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """사용 가능한 모델 리스트."""
    if model_server is None:
        raise HTTPException(status_code=503, detail="모델 서버가 초기화되지 않음")
    
    return list(model_server.model_info.values())


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """단일 사용자를 위한 추천 생성."""
    if model_server is None:
        raise HTTPException(status_code=503, detail="모델 서버가 초기화되지 않음")
    
    try:
        result = model_server.get_recommendations(
            user_id=request.user_id,
            num_items=request.num_items,
            filter_purchased=request.filter_purchased,
            include_scores=request.include_scores
        )
        return RecommendationResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"추천 생성 오류: {e}")
        raise HTTPException(status_code=500, detail="내부 서버 오류")


@app.post("/recommend/batch", response_model=List[RecommendationResponse])
async def get_batch_recommendations(request: BatchRecommendationRequest):
    """여러 사용자를 위한 추천 생성."""
    if model_server is None:
        raise HTTPException(status_code=503, detail="모델 서버가 초기화되지 않음")
    
    try:
        results = model_server.get_batch_recommendations(
            user_ids=request.user_ids,
            num_items=request.num_items,
            filter_purchased=request.filter_purchased,
            include_scores=request.include_scores
        )
        
        # 응답 모델로 변환
        responses = []
        for result in results:
            if "error" not in result:
                responses.append(RecommendationResponse(**result))
        
        return responses
    
    except Exception as e:
        logger.error(f"배치 추천 생성 오류: {e}")
        raise HTTPException(status_code=500, detail="내부 서버 오류")


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations_by_id(
    user_id: str,
    num_items: int = Query(12, ge=1, le=100),
    model: str = Query("best", description="사용할 모델"),
    filter_purchased: bool = Query(True, description="구매한 아이템 필터링"),
    include_scores: bool = Query(False, description="점수 포함")
):
    """사용자 ID로 추천 생성."""
    if model_server is None:
        raise HTTPException(status_code=503, detail="모델 서버가 초기화되지 않음")
    
    try:
        result = model_server.get_recommendations(
            user_id=user_id,
            model_name=model,
            num_items=num_items,
            filter_purchased=filter_purchased,
            include_scores=include_scores
        )
        return RecommendationResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"추천 생성 오류: {e}")
        raise HTTPException(status_code=500, detail="내부 서버 오류")


def main():
    """서버 실행을 위한 메인 함수."""
    import argparse
    
    parser = argparse.ArgumentParser(description="H&M 추천 API 서버 실행")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="바인드할 호스트")
    parser.add_argument("--port", type=int, default=8000, help="바인드할 포트")
    parser.add_argument("--reload", action="store_true", help="자동 리로드 활성화")
    parser.add_argument("--data-dir", type=str, default="data", help="데이터 디렉토리")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints",
                       help="체크포인트 디렉토리")
    
    args = parser.parse_args()
    
    # 환경 변수 설정
    os.environ["DATA_DIR"] = args.data_dir
    os.environ["CHECKPOINT_DIR"] = args.checkpoint_dir
    
    # 서버 실행
    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()