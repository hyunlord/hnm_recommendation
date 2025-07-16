"""프로젝트를 위한 상수 및 구성 값."""

from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"
ARTICLES_PATH = DATA_DIR / "articles.csv"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
TRANSACTIONS_PATH = DATA_DIR / "transactions_train.csv"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
IMAGES_DIR = DATA_DIR / "images"

# 출력 경로
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# 데이터 상수
DATE_COLUMN = "t_dat"
USER_COLUMN = "customer_id"
ITEM_COLUMN = "article_id"
PRICE_COLUMN = "price"

# 모델 상수
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_BATCH_SIZE = 1024
DEFAULT_NUM_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.001

# 평가 상수
DEFAULT_K_VALUES = [5, 10, 12, 20]
PRIMARY_METRIC = "map@12"

# 재현 가능성을 위한 랜덤 시드
RANDOM_SEED = 42