"""프로젝트를 위한 로깅 구성."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_file: Path = None, level=logging.INFO) -> logging.Logger:
    """일관된 형식으로 로거 설정.
    
    Args:
        name: 로거 이름
        log_file: 로그 파일의 선택적 경로
        level: 로깅 레벨
        
    Returns:
        구성된 로거 인스턴스
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 기존 핸들러 제거
    logger.handlers = []
    
    # 포매터 생성
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (로그 파일이 지정된 경우)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """프로젝트 이름 접두사가 있는 로거 인스턴스 가져오기.
    
    Args:
        name: 모듈 이름
        
    Returns:
        로거 인스턴스
    """
    return logging.getLogger(f"hnm_recommendation.{name}")