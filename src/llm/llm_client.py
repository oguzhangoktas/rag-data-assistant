"""
Unified LLM Client with Multi-Provider Support
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Provides a unified interface for interacting with multiple LLM providers
(OpenAI, Anthropic) with automatic fallback, retry logic, and cost tracking.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from enum import Enum

from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic, Anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.utils.config_loader import get_config, get_settings
from src.utils.logger import get_logger, LLMLogger
from src.utils.exceptions import (
    LLMException,
    LLMRateLimitException,
    LLMTimeoutException,
    LLMAuthenticationException,
)

logger = get_logger("llm")
llm_logger = LLMLogger()


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMResponse:
    """Standardized LLM response across all providers."""
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    raw_response: Optional[Any] = None


@dataclass
class LLMRequest:
    """Standardized LLM request."""
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    system_message: Optional[str] = None
    stream: bool = False
    request_id: Optional[str] = None


class CostCalculator:
    """Calculates LLM API costs based on token usage."""
    
    # Pricing per 1000 tokens (as of 2024)
    PRICING = {
        "openai": {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "text-embedding-3-small": {"input": 0.00002, "output": 0},
            "text-embedding-3-large": {"input": 0.00013, "output": 0},
        },
        "anthropic": {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        },
    }
    
    @classmethod
    def calculate(
        cls,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """
        Calculate cost in USD for LLM usage.
        
        Args:
            provider: LLM provider name
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        provider_pricing = cls.PRICING.get(provider, {})
        model_pricing = provider_pricing.get(model, {"input": 0.01, "output": 0.03})
        
        input_cost = (prompt_tokens / 1000) * model_pricing["input"]
        output_cost = (completion_tokens / 1000) * model_pricing["output"]
        
        return round(input_cost + output_cost, 6)


class OpenAIProvider:
    """OpenAI LLM provider implementation."""
    
    def __init__(self, api_key: str, default_model: str = "gpt-4-turbo-preview"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            default_model: Default model to use
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.sync_client = OpenAI(api_key=api_key)
        self.default_model = default_model
        self.provider_name = "openai"
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a completion using OpenAI.
        
        Args:
            request: LLM request parameters
            
        Returns:
            Standardized LLM response
        """
        start_time = time.time()
        model = request.model or self.default_model
        request_id = request.request_id or str(uuid.uuid4())[:8]
        
        try:
            messages = []
            
            # Add system message if provided
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message})
            
            # Add conversation messages
            messages.extend(request.messages)
            
            llm_logger.log_request(
                provider=self.provider_name,
                model=model,
                prompt_tokens=0,  # Will be updated after response
                request_id=request_id,
            )
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            cost_usd = CostCalculator.calculate(
                provider=self.provider_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            
            llm_logger.log_response(
                provider=self.provider_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                request_id=request_id,
                success=True,
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                provider=self.provider_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                finish_reason=response.choices[0].finish_reason,
                request_id=request_id,
                raw_response=response,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            llm_logger.log_response(
                provider=self.provider_name,
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=latency_ms,
                cost_usd=0,
                request_id=request_id,
                success=False,
                error=error_msg,
            )
            
            # Map to specific exceptions
            if "rate_limit" in error_msg.lower():
                raise LLMRateLimitException(
                    message=error_msg,
                    provider=self.provider_name,
                )
            elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                raise LLMAuthenticationException(
                    message=error_msg,
                    provider=self.provider_name,
                )
            elif "timeout" in error_msg.lower():
                raise LLMTimeoutException(
                    message=error_msg,
                    provider=self.provider_name,
                )
            else:
                raise LLMException(
                    message=error_msg,
                    provider=self.provider_name,
                    model=model,
                )

    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion using OpenAI.
        
        Args:
            request: LLM request parameters
            
        Yields:
            Content chunks as they arrive
        """
        model = request.model or self.default_model
        
        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        messages.extend(request.messages)
        
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider:
    """Anthropic (Claude) LLM provider implementation."""
    
    def __init__(self, api_key: str, default_model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            default_model: Default model to use
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.sync_client = Anthropic(api_key=api_key)
        self.default_model = default_model
        self.provider_name = "anthropic"
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a completion using Anthropic Claude.
        
        Args:
            request: LLM request parameters
            
        Returns:
            Standardized LLM response
        """
        start_time = time.time()
        model = request.model or self.default_model
        request_id = request.request_id or str(uuid.uuid4())[:8]
        
        try:
            # Convert messages to Anthropic format
            messages = []
            system_message = request.system_message or ""
            
            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })
            
            llm_logger.log_request(
                provider=self.provider_name,
                model=model,
                prompt_tokens=0,
                request_id=request_id,
            )
            
            response = await self.client.messages.create(
                model=model,
                max_tokens=request.max_tokens,
                system=system_message if system_message else None,
                messages=messages,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens
            
            cost_usd = CostCalculator.calculate(
                provider=self.provider_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            
            llm_logger.log_response(
                provider=self.provider_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                request_id=request_id,
                success=True,
            )
            
            # Extract text content
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
            
            return LLMResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                finish_reason=response.stop_reason,
                request_id=request_id,
                raw_response=response,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            llm_logger.log_response(
                provider=self.provider_name,
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=latency_ms,
                cost_usd=0,
                request_id=request_id,
                success=False,
                error=error_msg,
            )
            
            if "rate_limit" in error_msg.lower():
                raise LLMRateLimitException(
                    message=error_msg,
                    provider=self.provider_name,
                )
            elif "authentication" in error_msg.lower():
                raise LLMAuthenticationException(
                    message=error_msg,
                    provider=self.provider_name,
                )
            else:
                raise LLMException(
                    message=error_msg,
                    provider=self.provider_name,
                    model=model,
                )

    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion using Anthropic Claude.
        
        Args:
            request: LLM request parameters
            
        Yields:
            Content chunks as they arrive
        """
        model = request.model or self.default_model
        
        messages = []
        system_message = request.system_message or ""
        
        for msg in request.messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        
        async with self.client.messages.stream(
            model=model,
            max_tokens=request.max_tokens,
            system=system_message if system_message else None,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class LLMClient:
    """
    Unified LLM client with multi-provider support and automatic fallback.
    
    Features:
    - Multiple provider support (OpenAI, Anthropic)
    - Automatic fallback on failure
    - Retry logic with exponential backoff
    - Cost tracking
    - Streaming support
    
    Usage:
        client = LLMClient()
        response = await client.generate("What is the capital of France?")
    """
    
    def __init__(
        self,
        primary_provider: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        """
        Initialize the LLM client.
        
        Args:
            primary_provider: Primary provider name (openai/anthropic)
            fallback_provider: Fallback provider name
            openai_api_key: OpenAI API key (or from env)
            anthropic_api_key: Anthropic API key (or from env)
        """
        settings = get_settings()
        config = get_config().get_llm_config()
        
        self.primary_provider = primary_provider or settings.llm_primary_provider
        self.fallback_provider = fallback_provider or "anthropic"
        
        # Initialize providers
        self.providers: Dict[str, Union[OpenAIProvider, AnthropicProvider]] = {}
        
        openai_key = openai_api_key or settings.openai_api_key
        if openai_key:
            self.providers["openai"] = OpenAIProvider(
                api_key=openai_key,
                default_model=config.get("providers", {}).get("openai", {}).get("models", {}).get("default", "gpt-4-turbo-preview"),
            )
        
        anthropic_key = anthropic_api_key or settings.anthropic_api_key
        if anthropic_key:
            self.providers["anthropic"] = AnthropicProvider(
                api_key=anthropic_key,
                default_model=config.get("providers", {}).get("anthropic", {}).get("models", {}).get("default", "claude-3-sonnet-20240229"),
            )
        
        if not self.providers:
            raise LLMException(
                message="No LLM providers configured. Please set API keys.",
                error_code="NO_PROVIDERS",
            )
        
        # Fallback settings
        self.enable_fallback = config.get("fallback", {}).get("enabled", True)
        self.max_retries = 3
        
        logger.info(
            f"LLMClient initialized with primary={self.primary_provider}, "
            f"fallback={self.fallback_provider}, providers={list(self.providers.keys())}"
        )
    
    def _get_provider(self, provider_name: str) -> Union[OpenAIProvider, AnthropicProvider]:
        """Get a provider by name."""
        if provider_name not in self.providers:
            raise LLMException(
                message=f"Provider '{provider_name}' not configured",
                error_code="PROVIDER_NOT_FOUND",
            )
        return self.providers[provider_name]
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        provider: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a completion from an LLM.
        
        Args:
            prompt: User prompt/question
            system_message: Optional system message
            model: Optional model override
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            provider: Optional provider override
            request_id: Optional request ID for tracking
            
        Returns:
            Standardized LLM response
        """
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
            request_id=request_id or str(uuid.uuid4())[:8],
        )
        
        provider_name = provider or self.primary_provider
        
        try:
            llm_provider = self._get_provider(provider_name)
            return await llm_provider.generate(request)
            
        except LLMException as e:
            # Try fallback if enabled
            if self.enable_fallback and self.fallback_provider in self.providers:
                logger.warning(
                    f"Primary provider {provider_name} failed, trying fallback {self.fallback_provider}"
                )
                try:
                    fallback = self._get_provider(self.fallback_provider)
                    return await fallback.generate(request)
                except LLMException as fallback_error:
                    logger.error(f"Fallback provider also failed: {fallback_error}")
                    raise fallback_error
            raise e
    
    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        provider: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a completion from a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_message: Optional system message
            model: Optional model override
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            provider: Optional provider override
            request_id: Optional request ID for tracking
            
        Returns:
            Standardized LLM response
        """
        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
            request_id=request_id or str(uuid.uuid4())[:8],
        )
        
        provider_name = provider or self.primary_provider
        llm_provider = self._get_provider(provider_name)
        
        return await llm_provider.generate(request)
    
    async def generate_stream(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        provider: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion.
        
        Args:
            prompt: User prompt/question
            system_message: Optional system message
            model: Optional model override
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            provider: Optional provider override
            
        Yields:
            Content chunks as they arrive
        """
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
            stream=True,
        )
        
        provider_name = provider or self.primary_provider
        llm_provider = self._get_provider(provider_name)
        
        async for chunk in llm_provider.generate_stream(request):
            yield chunk


# Global singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    Get the global LLM client instance.
    
    Returns:
        LLMClient singleton instance
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
