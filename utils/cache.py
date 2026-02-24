"""
Result Caching System
Cache evaluation results to avoid redundant API calls
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class ResultCache:
    """Cache system for evaluation results"""
    
    def __init__(self, cache_dir: str = ".cache/eval_results"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(
        self,
        model_name: str,
        prompt: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key from model, prompt, and params"""
        content = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "params": params
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(
        self,
        model_name: str,
        prompt: str,
        params: Dict[str, Any],
        max_age_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result if exists and not expired
        
        Args:
            model_name: Name of the model
            prompt: The prompt/question
            params: Generation parameters
            max_age_hours: Maximum age of cache in hours
        
        Returns:
            Cached result or None if not found/expired
        """
        cache_key = self._get_cache_key(model_name, prompt, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check age
            cached_time = datetime.fromisoformat(cached['timestamp'])
            age = datetime.now() - cached_time
            
            if age > timedelta(hours=max_age_hours):
                return None
            
            return cached['result']
        except Exception:
            return None
    
    def set(
        self,
        model_name: str,
        prompt: str,
        params: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """
        Save result to cache
        
        Args:
            model_name: Name of the model
            prompt: The prompt/question
            params: Generation parameters
            result: Result to cache
        """
        cache_key = self._get_cache_key(model_name, prompt, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "result": result
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Silently fail on cache write errors
            pass
    
    def clear(self, older_than_hours: Optional[int] = None):
        """
        Clear cache
        
        Args:
            older_than_hours: Only clear entries older than this (optional)
        """
        if older_than_hours is None:
            # Clear all
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        else:
            # Clear old entries
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                    cached_time = datetime.fromisoformat(cached['timestamp'])
                    if cached_time < cutoff:
                        cache_file.unlink()
                except Exception:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "total_entries": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
