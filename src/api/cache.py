"""API를 위한 캐싱 유틸리티."""
import json
import time
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod
import redis
from functools import lru_cache
import hashlib


class CacheInterface(ABC):
    """추상 캐시 인터페이스."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """TTL과 함께 캐시에 값 설정."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """캐시에서 값 삭제."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """모든 캐시 지우기."""
        pass


class RedisCache(CacheInterface):
    """Redis 기반 캐시 구현."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """Redis 캐시 초기화.
        
        Args:
            host: Redis 호스트
            port: Redis 포트
            db: Redis 데이터베이스 번호
        """
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.prefix = "hnm_rec:"
    
    def _make_key(self, key: str) -> str:
        """접두사와 함께 캐시 키 생성."""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기."""
        try:
            value = self.client.get(self._make_key(key))
            if value:
                return json.loads(value)
        except Exception:
            pass
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """TTL과 함께 캐시에 값 설정."""
        try:
            self.client.setex(
                self._make_key(key),
                ttl,
                json.dumps(value)
            )
        except Exception:
            pass
    
    def delete(self, key: str) -> None:
        """캐시에서 값 삭제."""
        try:
            self.client.delete(self._make_key(key))
        except Exception:
            pass
    
    def clear(self) -> None:
        """모든 캐시 지우기."""
        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
        except Exception:
            pass


class InMemoryCache(CacheInterface):
    """메모리 내 캐시 구현."""
    
    def __init__(self, max_size: int = 1000):
        """메모리 내 캐시 초기화.
        
        Args:
            max_size: 최대 캐시 크기
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기."""
        if key in self.cache:
            entry = self.cache[key]
            if entry['expires_at'] > time.time():
                return entry['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """TTL과 함께 캐시에 값 설정."""
        # 캐시가 가듍 차면 가장 오래된 항목 제거
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['expires_at'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl
        }
    
    def delete(self, key: str) -> None:
        """캐시에서 값 삭제."""
        self.cache.pop(key, None)
    
    def clear(self) -> None:
        """모든 캐시 지우기."""
        self.cache.clear()


def make_cache_key(prefix: str, **kwargs) -> str:
    """매개변수로부터 캐시 키 생성.
    
    Args:
        prefix: 키 접두사
        **kwargs: 키에 포함할 매개변수
        
    Returns:
        캐시 키
    """
    # 일관된 키 생성을 위해 kwargs 정렬
    sorted_params = sorted(kwargs.items())
    param_str = json.dumps(sorted_params, sort_keys=True)
    
    # 긴 키를 위한 해시 생성
    if len(param_str) > 100:
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"{prefix}:{param_hash}"
    else:
        return f"{prefix}:{param_str}"