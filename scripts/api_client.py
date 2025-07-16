"""Example client for H&M Recommendation API."""
import requests
import json
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime


class RecommendationClient:
    """Client for H&M Recommendation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> Dict:
        """Check API health.
        
        Returns:
            Health status
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> List[Dict]:
        """List available models.
        
        Returns:
            List of model information
        """
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def get_recommendations(
        self,
        user_id: str,
        num_items: int = 12,
        model: str = "best",
        filter_purchased: bool = True,
        include_scores: bool = False
    ) -> Dict:
        """Get recommendations for a user.
        
        Args:
            user_id: User ID
            num_items: Number of items to recommend
            model: Model to use
            filter_purchased: Filter out purchased items
            include_scores: Include recommendation scores
            
        Returns:
            Recommendation response
        """
        # Using GET endpoint
        params = {
            "num_items": num_items,
            "model": model,
            "filter_purchased": filter_purchased,
            "include_scores": include_scores
        }
        
        response = requests.get(
            f"{self.base_url}/recommend/{user_id}",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_batch_recommendations(
        self,
        user_ids: List[str],
        num_items: int = 12,
        filter_purchased: bool = True,
        include_scores: bool = False
    ) -> List[Dict]:
        """Get recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            num_items: Number of items per user
            filter_purchased: Filter out purchased items
            include_scores: Include recommendation scores
            
        Returns:
            List of recommendation responses
        """
        data = {
            "user_ids": user_ids,
            "num_items": num_items,
            "filter_purchased": filter_purchased,
            "include_scores": include_scores
        }
        
        response = requests.post(
            f"{self.base_url}/recommend/batch",
            json=data
        )
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the API client."""
    # Initialize client
    client = RecommendationClient()
    
    print("H&M Recommendation API Client Example")
    print("=" * 50)
    
    # 1. Health check
    print("\n1. Health Check:")
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Models loaded: {health['models_loaded']}")
        print(f"Available models: {health['available_models']}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the API server is running!")
        return
    
    # 2. List models
    print("\n2. Available Models:")
    models = client.list_models()
    for model in models:
        print(f"- {model['name']} ({model['type']})")
        if model.get('metrics'):
            print(f"  MAP@12: {model['metrics'].get('test_map', 'N/A')}")
    
    # 3. Get recommendations for a single user
    print("\n3. Single User Recommendation:")
    user_id = "0"  # Example user ID
    try:
        recs = client.get_recommendations(
            user_id=user_id,
            num_items=5,
            include_scores=True
        )
        
        print(f"User: {recs['user_id']}")
        print(f"Model: {recs['model_name']}")
        print("Recommendations:")
        for i, item in enumerate(recs['recommendations'], 1):
            print(f"{i}. Article {item['article_id']}")
            if item.get('product_name'):
                print(f"   Name: {item['product_name']}")
            if item.get('score') is not None:
                print(f"   Score: {item['score']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 4. Batch recommendations
    print("\n4. Batch Recommendations:")
    user_ids = ["0", "1", "2", "3", "4"]  # Example user IDs
    try:
        batch_recs = client.get_batch_recommendations(
            user_ids=user_ids,
            num_items=3
        )
        
        print(f"Got recommendations for {len(batch_recs)} users")
        for rec in batch_recs[:2]:  # Show first 2
            print(f"\nUser {rec['user_id']}:")
            for item in rec['recommendations']:
                print(f"  - {item['article_id']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 5. Performance test
    print("\n5. Performance Test:")
    import time
    
    # Test single recommendations
    start = time.time()
    for i in range(10):
        client.get_recommendations(str(i))
    single_time = time.time() - start
    print(f"10 single requests: {single_time:.2f}s ({single_time/10:.3f}s per request)")
    
    # Test batch
    start = time.time()
    client.get_batch_recommendations([str(i) for i in range(10)])
    batch_time = time.time() - start
    print(f"1 batch request (10 users): {batch_time:.2f}s")
    print(f"Speedup: {single_time/batch_time:.1f}x")


if __name__ == "__main__":
    main()