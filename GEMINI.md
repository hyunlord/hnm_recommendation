# H&M 개인화 패션 추천 시스템 프로젝트

## 프로젝트 개요

H&M Kaggle 대회의 개인화 패션 추천 시스템 데이터를 활용하여, 사용자별 맞춤 상품 추천 시스템을 end-to-end로 구현한다. 이 프로젝트의 목표는 실제 Kaggle 대회 목표와 동일하게 이용자의 과거 거래 내역과 상품 메타데이터를 바탕으로 향후 1주일 간 구매할 가능성이 높은 상위 12개 상품을 예측 및 추천하는 것이다. 추천 결과의 평가는 MAP@12 (Mean Average Precision @ 12) 척도를 사용하며, 이는 상위 12개 추천 목록에서 실제 구매한 상품들의 순위 관련 정확도를 측정하는 지표이다.

## 프로젝트 단계별 계획

### 1단계: 프로젝트 셋업 및 데이터 준비
- GitHub 레포지토리 생성 (hm-recommendation 등)
- README에 목표, 데이터 개요, 전체 구조 설명 추가
- data/, notebooks/, src/ 폴더 구조 설계
- Kaggle API 또는 수동으로 데이터 다운로드
- 주요 파일 정리: transactions.csv, articles.csv, customers.csv
- 데이터 요약/EDA 노트북 작성 (notebooks/eda.ipynb)

### 2단계: PyTorch Lightning + DataModule 설정
- DataModule에서 user, item index 매핑 및 저장
- 거래 로그를 기준으로 train/val/test split (시간 기준)
- get_train_dataloader, get_val_dataloader 구현

### 3단계: Baseline 추천 알고리즘 구현
- 가장 간단한 Popular Items 추천 (전체 Top-N)
- UserCF, ItemCF (Surprise 또는 직접 구현)
- Matrix Factorization (e.g. ALS 또는 PyTorch 구현)
- MAP@12 등 지표 측정 및 baseline 기록

### 4단계: Neural CF (NeuMF) 모델 구현
- LightningModule으로 모델 클래스 정의
- 사용자/아이템 임베딩 + MLP 구조
- BPR Loss or BCE Loss
- Negative Sampling 로직 구현
- Validation 및 MAP@12 로그 저장

### 5단계: Content-Based / Hybrid 모델
- 상품 메타데이터 전처리 (카테고리, 색상, 텍스트 등)
- TF-IDF 벡터화 또는 Embedding
- 유사도 기반 콘텐츠 추천 + 유저 프로파일 생성
- 콘텐츠 기반 + CF score ensemble 구현

### 6단계: LightGCN, SASRec 등 심화 모델
- LightGCN: PyTorch or PyG 기반 구현
- SASRec: Transformer 기반 순차 추천 모델

### 7단계: 모델 평가 및 비교
- MAP@12, Precision@12, NDCG@12 비교 테이블
- 각 추천 결과 예시 시각화 및 분석
- best model 선정

### 8단계: 모델 서빙 및 MLOps
- FastAPI 기반 REST API 서버 구축
- GET /recommend?user_id=xxx 형태로 결과 반환
- Dockerfile + docker-compose 배포
- (선택) MLflow로 실험 관리, DVC로 데이터 버전 관리
- (선택) Streamlit 데모 페이지 구축
