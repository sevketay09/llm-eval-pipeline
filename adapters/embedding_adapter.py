"""
Unified Embedding Adapter
Supports: HuggingFace (local), LiteLLM (remote), vLLM (remote), OpenAI-compatible APIs
"""
import time
import httpx
import numpy as np
from typing import List, Dict, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedEmbeddingAdapter:
    """Unified interface for all embedding model providers"""
    
    def __init__(self, config: Dict[str, Any], model_key: Optional[str] = None):
        self.config = config
        self.model_key = model_key or config.get("model_key", "unknown")
        self.provider = config.get("provider", "huggingface")
        self.model_name = config["model_name"]
        self.embedding_dim = config.get("embedding_dim", None)
        self.max_sequence_length = config.get("max_sequence_length", 512)
        
        logger.info(f"Initializing embedding adapter for {self.model_name} (provider: {self.provider})")
        
        # Initialize appropriate client
        if self.provider == "huggingface":
            self._init_huggingface()
        elif self.provider in ["litellm", "vllm", "openai"]:
            self._init_api_client()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        
        # Performance tracking
        self.total_embeddings = 0
        self.latencies = []
        self.error_count = 0
    
    def _init_huggingface(self):
        """Initialize HuggingFace local model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_path = self.config.get("model_path", self.model_name)
            logger.info(f"Loading HuggingFace model from {model_path}")
            
            self.model = SentenceTransformer(model_path)
            
            # Auto-detect embedding dimension if not specified
            if self.embedding_dim is None:
                test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                self.embedding_dim = test_embedding.shape[1]
                logger.info(f"Auto-detected embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {str(e)}")
            raise
    
    def _init_api_client(self):
        """Initialize API client for LiteLLM/vLLM/OpenAI"""
        self.api_url = self.config.get("base_url", "").rstrip("/") + "/v1/embeddings"
        self.api_key = self.config.get("api_key", "")
        self.timeout = self.config.get("timeout", 60.0)
        
        logger.info(f"API endpoint: {self.api_url}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Generate embeddings for input texts
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing (HuggingFace only)
            normalize: Whether to normalize embeddings (L2 norm)
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        start_time = time.time()
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            if self.provider == "huggingface":
                embeddings = self._encode_huggingface(texts, batch_size, normalize, **kwargs)
            else:
                embeddings = self._encode_api(texts, normalize, **kwargs)
            
            latency = time.time() - start_time
            self.latencies.append(latency)
            self.total_embeddings += len(texts)
            
            logger.debug(f"Generated {len(texts)} embeddings in {latency:.3f}s")
            
            return {
                'embeddings': embeddings,
                'latency': latency,
                'model': self.model_name,
                'count': len(texts)
            }
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def _encode_huggingface(
        self,
        texts: List[str],
        batch_size: Optional[int],
        normalize: bool,
        **kwargs
    ) -> np.ndarray:
        """Encode using local HuggingFace model"""
        batch_size = batch_size or self.config.get("batch_size", 32)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
            **kwargs
        )
        
        return embeddings
    
    def _encode_api(
        self,
        texts: List[str],
        normalize: bool,
        **kwargs
    ) -> np.ndarray:
        """Encode using API (LiteLLM/vLLM/OpenAI)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "input": texts
        }
        
        # Add any extra parameters
        data.update(kwargs)
        
        try:
            response = httpx.post(
                url=self.api_url,
                json=data,
                headers=headers,
                timeout=self.timeout,
                verify=False  # For internal SSL certificates
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract embeddings from response
            embeddings_list = []
            for item in result["data"]:
                embeddings_list.append(item["embedding"])
            
            embeddings = np.array(embeddings_list, dtype=np.float32)
            
            # Auto-detect embedding dimension
            if self.embedding_dim is None:
                self.embedding_dim = embeddings.shape[1]
                logger.info(f"Auto-detected embedding dimension: {self.embedding_dim}")
            
            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)
            
            return embeddings
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between two sets of embeddings
        
        Args:
            embeddings1: Array of shape (n, dim)
            embeddings2: Array of shape (m, dim)
            metric: "cosine" or "dot"
            
        Returns:
            Similarity matrix of shape (n, m)
        """
        if metric == "cosine":
            # Cosine similarity (assumes normalized embeddings)
            return np.dot(embeddings1, embeddings2.T)
        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)
        elif metric == "euclidean":
            # Negative euclidean distance (higher = more similar)
            from scipy.spatial.distance import cdist
            return -cdist(embeddings1, embeddings2, metric='euclidean')
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "total_embeddings": self.total_embeddings,
            "error_count": self.error_count,
            "avg_latency": np.mean(self.latencies) if self.latencies else 0,
            "p95_latency": np.percentile(self.latencies, 95) if self.latencies else 0,
            "embedding_dim": self.embedding_dim
        }
    
    def reset_stats(self):
        """Reset tracking statistics"""
        self.total_embeddings = 0
        self.latencies = []
        self.error_count = 0
