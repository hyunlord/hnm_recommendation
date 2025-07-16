# H&M 추천 API

## 개요

H&M 추천 API는 학습된 모델을 기반으로 개인화된 패션 추천을 제공하는 RESTful 엔드포인트를 제공합니다.

## 주요 기능

- **다중 모델 지원**: 다양한 학습된 모델로부터 추천 제공
- **배치 처리**: 단일 요청으로 여러 사용자에 대한 추천 획득
- **필터링**: 이미 구매한 상품 필터링 옵션
- **캐싱**: 성능 향상을 위한 내장 캐싱 지원
- **헬스 모니터링**: 서비스 모니터링을 위한 헬스 체크 엔드포인트
- **Docker 지원**: Docker 및 docker-compose를 통한 간편한 배포

## 빠른 시작

### 1. 로컬 개발

```bash
# 의존성 설치
pip install fastapi uvicorn redis

# 서버 실행
python scripts/serve.py --host 0.0.0.0 --port 8000

# 특정 모델 디렉토리 지정
python scripts/serve.py --checkpoint-dir experiments/best_model/checkpoints
```

### 2. Docker 배포

```bash
# docker-compose로 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f recommendation-api

# 서비스 중지
docker-compose down
```

### 3. API 사용

```python
# Python 클라이언트 사용 예제
from scripts.api_client import RecommendationClient

client = RecommendationClient("http://localhost:8000")

# 사용자에 대한 추천 받기
recs = client.get_recommendations(
    user_id="12345",
    num_items=10,
    include_scores=True
)

print(f"사용자 {recs['user_id']}에 대한 추천:")
for item in recs['recommendations']:
    print(f"- {item['article_id']}: {item['product_name']}")
```

## API 엔드포인트

### 헬스 체크
```
GET /health
```
API 상태 및 로드된 모델 정보를 반환합니다.

### 모델 목록 조회
```
GET /models
```
사용 가능한 모델에 대한 정보를 반환합니다.

### 추천 받기
```
GET /recommend/{user_id}?num_items=12&model=best&filter_purchased=true
```

매개변수:
- `user_id`: 사용자 ID (필수)
- `num_items`: 추천할 상품 수 (기본값: 12)
- `model`: 사용할 모델 (기본값: "best")
- `filter_purchased`: 구매한 상품 필터링 여부 (기본값: true)
- `include_scores`: 추천 점수 포함 여부 (기본값: false)

### 배치 추천
```
POST /recommend/batch
```

요청 본문:
```json
{
  "user_ids": ["user1", "user2", "user3"],
  "num_items": 12,
  "filter_purchased": true,
  "include_scores": false
}
```

## 설정

### 환경 변수

- `DATA_DIR`: 데이터 디렉토리 경로 (기본값: "data")
- `CHECKPOINT_DIR`: 모델 체크포인트 경로 (기본값: "experiments/checkpoints")
- `DEVICE`: 추론 장치 (기본값: 가용시 "cuda", 아니면 "cpu")
- `REDIS_HOST`: 캐싱용 Redis 호스트 (기본값: "localhost")
- `REDIS_PORT`: Redis 포트 (기본값: 6379)

### 모델 선택

API는 체크포인트 디렉토리에서 사용 가능한 모든 모델을 자동으로 로드합니다. "best" 모델은 검증 메트릭(MAP@12) 기준으로 선택됩니다.

## 성능 최적화

### 1. 캐싱

API는 두 가지 캐싱 백엔드를 지원합니다:
- **Redis**: 프로덕션 배포용
- **인메모리**: 개발/단일 인스턴스 배포용

캐시 TTL은 엔드포인트별로 설정 가능합니다.

### 2. 배치 처리

여러 사용자에 대해서는 배치 엔드포인트를 사용하여 오버헤드를 줄이세요:
```python
# 느림: 여러 개별 요청
for user_id in user_ids:
    client.get_recommendations(user_id)

# 빠름: 단일 배치 요청
client.get_batch_recommendations(user_ids)
```

### 3. 모델 로딩

모델은 시작 시 한 번 로드되어 빠른 추론을 위해 메모리에 유지됩니다.

## 모니터링

### 메트릭

API는 다음 메트릭을 노출합니다:
- 요청 수 및 지연 시간
- 캐시 적중/미스 비율
- 모델 추론 시간
- 오류율

### 로깅

설정 가능한 레벨의 구조화된 로깅:
```bash
# 디버그 로깅으로 실행
python scripts/serve.py --log-level debug
```

## 프로덕션 배포

### 1. Gunicorn 사용

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 60 \
    scripts.serve:app
```

### 2. Nginx와 함께

제공된 `nginx.conf`를 다음 용도로 사용:
- 로드 밸런싱
- SSL 종료
- 응답 캐싱
- 속도 제한

### 3. Kubernetes

배포 예제:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hnm-recommendation-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hnm-api
  template:
    metadata:
      labels:
        app: hnm-api
    spec:
      containers:
      - name: api
        image: hnm-recommendation-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEVICE
          value: "cpu"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## 테스트

### 단위 테스트
```bash
pytest tests/test_api.py
```

### 부하 테스트
```bash
# locust 사용
locust -f tests/load_test.py --host http://localhost:8000
```

### 통합 테스트
```bash
# 모든 엔드포인트 테스트
python scripts/test_api_integration.py
```

## 문제 해결

### 일반적인 문제

1. **모델 로딩 실패**: 체크포인트 경로 및 모델 호환성 확인
2. **메모리 부족**: 배치 크기 줄이기 또는 GPU 대신 CPU 사용
3. **느린 응답**: 캐싱 활성화 및 모델 복잡도 확인
4. **사용자를 찾을 수 없음**: 처리된 데이터에 사용자가 존재하는지 확인

### 디버그 모드

디버그 로깅으로 실행:
```bash
python scripts/serve.py --reload --log-level debug
```

## API 클라이언트 예제

### Python
```python
import requests

# 추천 받기
response = requests.get(
    "http://localhost:8000/recommend/12345",
    params={"num_items": 5}
)
recommendations = response.json()
```

### cURL
```bash
# 추천 받기
curl "http://localhost:8000/recommend/12345?num_items=5"

# 배치 추천
curl -X POST "http://localhost:8000/recommend/batch" \
  -H "Content-Type: application/json" \
  -d '{"user_ids": ["123", "456"], "num_items": 5}'
```

### JavaScript
```javascript
// 추천 받기
fetch('http://localhost:8000/recommend/12345?num_items=5')
  .then(response => response.json())
  .then(data => console.log(data));
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.