"""
Token Counter for LLM Cost Estimation
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Provides accurate token counting for different LLM providers using
tiktoken for OpenAI models and character-based estimation for others.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from src.utils.logger import get_logger

logger = get_logger("token_counter")


@dataclass
class TokenCount:
    """Token count result with metadata."""
    text_tokens: int
    estimated: bool
    model: str
    encoding: str


@dataclass
class CostEstimate:
    """Cost estimation result."""
    prompt_tokens: int
    estimated_completion_tokens: int
    total_tokens: int
    input_cost_usd: float
    estimated_output_cost_usd: float
    total_estimated_cost_usd: float
    model: str


class TokenCounter:
    """
    Counts tokens for LLM inputs and outputs.
    
    Features:
    - Accurate token counting using tiktoken for OpenAI models
    - Character-based estimation for non-OpenAI models
    - Cost estimation based on token counts
    - Caching for performance
    
    Usage:
        counter = TokenCounter()
        count = counter.count_tokens("Hello, world!", model="gpt-4")
        cost = counter.estimate_cost("Hello, world!", model="gpt-4")
    """
    
    # Model to encoding mapping for OpenAI
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
    }
    
    # Pricing per 1000 tokens in USD
    PRICING = {
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "text-embedding-3-small": {"input": 0.00002, "output": 0},
        "text-embedding-3-large": {"input": 0.00013, "output": 0},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    }
    
    # Default completion ratio (completion tokens / prompt tokens)
    DEFAULT_COMPLETION_RATIO = 0.5
    
    def __init__(self):
        """Initialize the token counter."""
        self._encoders: Dict[str, tiktoken.Encoding] = {}
    
    @lru_cache(maxsize=10)
    def _get_encoder(self, encoding_name: str) -> tiktoken.Encoding:
        """
        Get or create a tiktoken encoder.
        
        Args:
            encoding_name: Name of the encoding
            
        Returns:
            Tiktoken encoder instance
        """
        return tiktoken.get_encoding(encoding_name)
    
    def _get_encoding_for_model(self, model: str) -> Optional[str]:
        """
        Get the encoding name for a model.
        
        Args:
            model: Model name
            
        Returns:
            Encoding name or None if not found
        """
        return self.MODEL_ENCODINGS.get(model)
    
    def count_tokens(
        self,
        text: Union[str, List[str]],
        model: str = "gpt-4-turbo-preview",
    ) -> TokenCount:
        """
        Count tokens in text for a specific model.
        
        Args:
            text: Text or list of texts to count tokens for
            model: Model name for encoding selection
            
        Returns:
            TokenCount with token count and metadata
        """
        if isinstance(text, list):
            text = " ".join(text)
        
        encoding_name = self._get_encoding_for_model(model)
        
        if encoding_name:
            # Use tiktoken for accurate counting
            try:
                encoder = self._get_encoder(encoding_name)
                tokens = encoder.encode(text)
                return TokenCount(
                    text_tokens=len(tokens),
                    estimated=False,
                    model=model,
                    encoding=encoding_name,
                )
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed, using estimation: {e}")
        
        # Fallback to character-based estimation
        # Rough rule: ~4 characters per token for English text
        estimated_tokens = len(text) // 4
        
        return TokenCount(
            text_tokens=estimated_tokens,
            estimated=True,
            model=model,
            encoding="character_estimate",
        )
    
    def count_messages_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4-turbo-preview",
    ) -> int:
        """
        Count tokens in a list of chat messages.
        
        Accounts for message formatting overhead.
        
        Args:
            messages: List of message dictionaries
            model: Model name for encoding selection
            
        Returns:
            Total token count
        """
        total_tokens = 0
        
        # Base tokens per message (role, content delimiters)
        tokens_per_message = 4  # Approximate overhead per message
        
        for message in messages:
            # Count content tokens
            content = message.get("content", "")
            count = self.count_tokens(content, model)
            total_tokens += count.text_tokens
            
            # Add message overhead
            total_tokens += tokens_per_message
            
            # Add role tokens
            role = message.get("role", "user")
            role_count = self.count_tokens(role, model)
            total_tokens += role_count.text_tokens
        
        # Add conversation overhead
        total_tokens += 3  # Every reply is primed with assistant
        
        return total_tokens
    
    def estimate_cost(
        self,
        text: Union[str, List[str]],
        model: str = "gpt-4-turbo-preview",
        estimated_completion_tokens: Optional[int] = None,
        completion_ratio: float = DEFAULT_COMPLETION_RATIO,
    ) -> CostEstimate:
        """
        Estimate the cost for an LLM request.
        
        Args:
            text: Input text or list of texts
            model: Model name
            estimated_completion_tokens: Optional explicit completion token count
            completion_ratio: Ratio of completion to prompt tokens if not specified
            
        Returns:
            CostEstimate with detailed cost breakdown
        """
        token_count = self.count_tokens(text, model)
        prompt_tokens = token_count.text_tokens
        
        # Estimate completion tokens if not provided
        if estimated_completion_tokens is None:
            estimated_completion_tokens = int(prompt_tokens * completion_ratio)
        
        total_tokens = prompt_tokens + estimated_completion_tokens
        
        # Get pricing for model
        pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.03})
        
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (estimated_completion_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return CostEstimate(
            prompt_tokens=prompt_tokens,
            estimated_completion_tokens=estimated_completion_tokens,
            total_tokens=total_tokens,
            input_cost_usd=round(input_cost, 6),
            estimated_output_cost_usd=round(output_cost, 6),
            total_estimated_cost_usd=round(total_cost, 6),
            model=model,
        )
    
    def calculate_actual_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """
        Calculate actual cost after knowing token counts.
        
        Args:
            model: Model name
            prompt_tokens: Actual prompt tokens used
            completion_tokens: Actual completion tokens used
            
        Returns:
            Cost in USD
        """
        pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.03})
        
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        
        return round(input_cost + output_cost, 6)
    
    def truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int,
        model: str = "gpt-4-turbo-preview",
        truncation_indicator: str = "...[truncated]",
    ) -> str:
        """
        Truncate text to fit within a token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum allowed tokens
            model: Model name for encoding
            truncation_indicator: Text to append if truncated
            
        Returns:
            Truncated text
        """
        token_count = self.count_tokens(text, model)
        
        if token_count.text_tokens <= max_tokens:
            return text
        
        encoding_name = self._get_encoding_for_model(model)
        
        if encoding_name:
            try:
                encoder = self._get_encoder(encoding_name)
                tokens = encoder.encode(text)
                
                # Leave room for truncation indicator
                indicator_tokens = len(encoder.encode(truncation_indicator))
                truncated_tokens = tokens[: max_tokens - indicator_tokens]
                
                truncated_text = encoder.decode(truncated_tokens)
                return truncated_text + truncation_indicator
                
            except Exception as e:
                logger.warning(f"Token truncation failed, using character truncation: {e}")
        
        # Fallback to character-based truncation
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        
        if len(text) > max_chars:
            return text[: max_chars - len(truncation_indicator)] + truncation_indicator
        
        return text
    
    def get_model_context_limit(self, model: str) -> int:
        """
        Get the context window limit for a model.
        
        Args:
            model: Model name
            
        Returns:
            Maximum context tokens
        """
        context_limits = {
            "gpt-4-turbo-preview": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
        }
        return context_limits.get(model, 8192)
    
    def get_pricing_info(self, model: str) -> Dict[str, float]:
        """
        Get pricing information for a model.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with input and output pricing per 1000 tokens
        """
        return self.PRICING.get(model, {"input": 0.01, "output": 0.03})


# Convenience functions
def count_tokens(text: str, model: str = "gpt-4-turbo-preview") -> int:
    """
    Quick function to count tokens.
    
    Args:
        text: Text to count
        model: Model name
        
    Returns:
        Token count
    """
    counter = TokenCounter()
    return counter.count_tokens(text, model).text_tokens


def estimate_cost(text: str, model: str = "gpt-4-turbo-preview") -> float:
    """
    Quick function to estimate cost.
    
    Args:
        text: Input text
        model: Model name
        
    Returns:
        Estimated cost in USD
    """
    counter = TokenCounter()
    return counter.estimate_cost(text, model).total_estimated_cost_usd
