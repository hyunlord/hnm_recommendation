# H&M 개인화 패션 추천 시스템

이 프로젝트는 H&M Kaggle 경진대회 데이터셋을 사용하여 종단간(end-to-end) 개인화 패션 추천 시스템을 구현합니다. 협업 필터링, 콘텐츠 기반 필터링, 딥러닝 접근법 등 다양한 추천 알고리즘을 활용합니다.

## 프로젝트 구조

```
hnm_recommendation/
├── data/                      # 원본 데이터 디렉토리
│   ├── articles.csv           # 상품 메타데이터
│   ├── customers.csv          # 고객 정보
│   ├── transactions_train.csv # 거래 이력
│   └── images/                # 상품 이미지
├── src/                       # 소스 코드
│   ├── data/                  # 데이터 처리 모듈
│   ├── models/                # 추천 모델
│   ├── utils/                 # 유틸리티 함수
│   ├── api/                   # API 서빙 코드
│   └── evaluation/            # 평가 메트릭
├── notebooks/                 # EDA용 Jupyter 노트북
├── configs/                   # 설정 파일
├── experiments/               # 실험 결과물
│   ├── logs/                  # 학습 로그
│   ├── checkpoints/           # 모델 체크포인트
│   └── results/               # 평가 결과
├── scripts/                   # 실행 스크립트
├── tests/                     # 단위 테스트
└── docs/                      # 문서
```

## 주요 기능

- **다양한 추천 알고리즘**:
  - 협업 필터링 (사용자 기반, 아이템 기반, 행렬 분해)
  - 콘텐츠 기반 필터링
  - 딥러닝 모델 (Neural CF, Wide&Deep, LightGCN)
  - 순차 모델 (SASRec, GRU4Rec)

- **PyTorch Lightning 통합**: 모듈화되고 확장 가능한 모델 학습
- **실험 추적**: MLflow 및 Weights & Biases 지원
- **모델 서빙**: FastAPI 기반 REST API
- **MLOps 파이프라인**: 자동화된 학습 및 배포

## 설치 방법

1. 저장소 복제:
```bash
git clone https://github.com/yourusername/hnm_recommendation.git
cd hnm_recommendation
```

2. 가상환경 생성:
```bash
python -m venv venv
source venv/bin/activate  # Windows에서는: venv\Scripts\activate
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
pip install -e .  # 개발 모드로 패키지 설치
```

## 빠른 시작

1. **데이터 준비**:
   - Kaggle에서 H&M 데이터셋 다운로드
   - CSV 파일을 `data/` 디렉토리에 배치

2. **EDA 실행**:
```bash
jupyter lab notebooks/eda.ipynb
```

3. **모델 학습**:
```bash
python scripts/train.py model=neural_cf
```

4. **모델 평가**:
```bash
python scripts/evaluate.py
```

5. **API 서버 시작**:
```bash
python scripts/serve.py
```

## 모델 성능

| 모델 | MAP@12 | Recall@12 | Precision@12 |
|------|--------|-----------|--------------|
| TBD  | TBD    | TBD       | TBD          |

## 사용법

### 모델 학습

```python
from src.models import NeuralCF
from src.data import HMDataModule

# 데이터 모듈 초기화
data_module = HMDataModule(
    data_dir="data/",
    batch_size=1024,
    num_workers=4
)

# 모델 초기화
model = NeuralCF(
    num_users=data_module.num_users,
    num_items=data_module.num_items,
    embedding_dim=64
)

# PyTorch Lightning으로 학습
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, data_module)
```

### 예측 수행

```python
# 학습된 모델 로드
model = NeuralCF.load_from_checkpoint("path/to/checkpoint.ckpt")

# 사용자에 대한 추천 생성
user_id = "some_user_id"
recommendations = model.recommend(user_id, k=12)
```

## API 엔드포인트

- `GET /recommend/{user_id}`: 사용자에 대한 상위 12개 추천
- `GET /health`: 헬스 체크 엔드포인트
- `POST /feedback`: 사용자 피드백 제출

## 기여 방법

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다 - 자세한 내용은 LICENSE 파일을 참조하세요.

## 감사의 말

- Kaggle을 통해 데이터셋을 제공한 H&M Group
- 훌륭한 프레임워크를 제공한 PyTorch Lightning 팀
- 추천 시스템 연구 커뮤니티