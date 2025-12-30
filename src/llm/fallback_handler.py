"""
Fallback Handler for LLM Provider Failover
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Manages automatic failover between LLM providers when the primary
provider fails, with configurable retry logic and circuit breaker patterns.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar
from enum import Enum
from datetime import datetime, timedelta

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from src.utils.logger import get_logger
from src.utils.exceptions import (
    LLMException,
    LLMRateLimitException,
    LLMTimeoutException,
)

logger = get_logger("fallback_handler")

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class ProviderHealth:
    """Health status for an LLM provider."""
    provider_name: str
    is_healthy: bool = True
    consecutive_failures: int = 0
    total_failures: int = 0
    total_requests: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    circuit_state: CircuitState = CircuitState.CLOSED
    circuit_opened_at: Optional[datetime] = None
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.total_failures / self.total_requests
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available for requests."""
        if self.circuit_state == CircuitState.OPEN:
            return False
        return True


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 10.0
    exponential_base: float = 2.0
    failure_threshold: int = 5  # Failures before circuit opens
    recovery_timeout_seconds: int = 60  # Time before testing recovery
    half_open_requests: int = 3  # Test requests in half-open state


class FallbackHandler:
    """
    Handles LLM provider failover with circuit breaker pattern.
    
    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker to prevent cascading failures
    - Provider health tracking
    - Intelligent fallback selection
    
    Usage:
        handler = FallbackHandler(providers=["openai", "anthropic"])
        result = await handler.execute_with_fallback(
            primary_fn=openai_call,
            fallback_fn=anthropic_call,
        )
    """
    
    def __init__(
        self,
        providers: List[str],
        config: Optional[FallbackConfig] = None,
    ):
        """
        Initialize the fallback handler.
        
        Args:
            providers: List of provider names in priority order
            config: Optional fallback configuration
        """
        self.providers = providers
        self.config = config or FallbackConfig()
        
        # Initialize health tracking for each provider
        self._health: Dict[str, ProviderHealth] = {
            provider: ProviderHealth(provider_name=provider)
            for provider in providers
        }
        
        logger.info(f"FallbackHandler initialized with providers: {providers}")
    
    def _record_success(self, provider: str) -> None:
        """Record a successful request for a provider."""
        health = self._health.get(provider)
        if health:
            health.total_requests += 1
            health.consecutive_failures = 0
            health.last_success_time = datetime.now()
            health.is_healthy = True
            
            # If in half-open state and successful, close circuit
            if health.circuit_state == CircuitState.HALF_OPEN:
                health.circuit_state = CircuitState.CLOSED
                health.circuit_opened_at = None
                logger.info(f"Circuit closed for provider {provider}")
    
    def _record_failure(self, provider: str, error: Exception) -> None:
        """Record a failed request for a provider."""
        health = self._health.get(provider)
        if health:
            health.total_requests += 1
            health.total_failures += 1
            health.consecutive_failures += 1
            health.last_failure_time = datetime.now()
            
            logger.warning(
                f"Provider {provider} failure #{health.consecutive_failures}: {error}"
            )
            
            # Check if circuit should open
            if health.consecutive_failures >= self.config.failure_threshold:
                if health.circuit_state != CircuitState.OPEN:
                    health.circuit_state = CircuitState.OPEN
                    health.circuit_opened_at = datetime.now()
                    health.is_healthy = False
                    logger.warning(f"Circuit opened for provider {provider}")
    
    def _check_circuit(self, provider: str) -> bool:
        """
        Check if a provider's circuit allows requests.
        
        Args:
            provider: Provider name
            
        Returns:
            True if requests are allowed
        """
        health = self._health.get(provider)
        if not health:
            return True
        
        if health.circuit_state == CircuitState.CLOSED:
            return True
        
        if health.circuit_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if health.circuit_opened_at:
                elapsed = datetime.now() - health.circuit_opened_at
                if elapsed.total_seconds() >= self.config.recovery_timeout_seconds:
                    # Transition to half-open
                    health.circuit_state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit half-open for provider {provider}")
                    return True
            return False
        
        # Half-open state allows requests
        return True
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available providers.
        
        Returns:
            List of provider names that are available
        """
        return [
            provider for provider in self.providers
            if self._check_circuit(provider)
        ]
    
    def get_provider_health(self, provider: str) -> Optional[ProviderHealth]:
        """
        Get health status for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            ProviderHealth or None if not found
        """
        return self._health.get(provider)
    
    def get_all_health(self) -> Dict[str, ProviderHealth]:
        """
        Get health status for all providers.
        
        Returns:
            Dictionary of provider health statuses
        """
        return self._health.copy()
    
    async def execute_with_fallback(
        self,
        operations: Dict[str, Callable[[], Any]],
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Execute an operation with automatic fallback.
        
        Args:
            operations: Dictionary mapping provider name to async callable
            timeout: Optional timeout in seconds
            
        Returns:
            Result from successful operation
            
        Raises:
            LLMException: If all providers fail
        """
        available = self.get_available_providers()
        
        if not available:
            raise LLMException(
                message="No LLM providers available (all circuits open)",
                error_code="ALL_PROVIDERS_UNAVAILABLE",
            )
        
        errors = []
        
        for provider in available:
            if provider not in operations:
                continue
            
            operation = operations[provider]
            
            try:
                # Execute with retry logic
                result = await self._execute_with_retry(
                    provider=provider,
                    operation=operation,
                    timeout=timeout,
                )
                self._record_success(provider)
                return result
                
            except Exception as e:
                self._record_failure(provider, e)
                errors.append((provider, e))
                logger.warning(f"Provider {provider} failed, trying next...")
                continue
        
        # All providers failed
        error_msg = "; ".join([f"{p}: {e}" for p, e in errors])
        raise LLMException(
            message=f"All LLM providers failed: {error_msg}",
            error_code="ALL_PROVIDERS_FAILED",
            details={"errors": [str(e) for _, e in errors]},
        )
    
    async def _execute_with_retry(
        self,
        provider: str,
        operation: Callable[[], Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            provider: Provider name for tracking
            operation: Async callable to execute
            timeout: Optional timeout in seconds
            
        Returns:
            Result from successful operation
        """
        last_error = None
        delay = self.config.initial_delay_seconds
        
        for attempt in range(self.config.max_retries):
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        operation(),
                        timeout=timeout,
                    )
                else:
                    result = await operation()
                return result
                
            except asyncio.TimeoutError:
                last_error = LLMTimeoutException(
                    message=f"Operation timed out after {timeout}s",
                    provider=provider,
                    timeout_seconds=int(timeout) if timeout else None,
                )
            except LLMRateLimitException as e:
                last_error = e
                # Use longer delay for rate limits
                delay = min(delay * 2, self.config.max_delay_seconds)
            except LLMException as e:
                last_error = e
            except Exception as e:
                last_error = LLMException(
                    message=str(e),
                    provider=provider,
                )
            
            # Wait before retry (except on last attempt)
            if attempt < self.config.max_retries - 1:
                logger.debug(f"Retry {attempt + 1} for {provider} after {delay}s")
                await asyncio.sleep(delay)
                delay = min(
                    delay * self.config.exponential_base,
                    self.config.max_delay_seconds,
                )
        
        raise last_error or LLMException(
            message=f"Operation failed after {self.config.max_retries} retries",
            provider=provider,
        )
    
    def reset_circuit(self, provider: str) -> None:
        """
        Manually reset a provider's circuit breaker.
        
        Args:
            provider: Provider name
        """
        health = self._health.get(provider)
        if health:
            health.circuit_state = CircuitState.CLOSED
            health.circuit_opened_at = None
            health.consecutive_failures = 0
            health.is_healthy = True
            logger.info(f"Circuit manually reset for provider {provider}")
    
    def reset_all_circuits(self) -> None:
        """Reset all circuit breakers."""
        for provider in self.providers:
            self.reset_circuit(provider)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all providers.
        
        Returns:
            Dictionary with provider statistics
        """
        stats = {}
        for provider, health in self._health.items():
            stats[provider] = {
                "is_healthy": health.is_healthy,
                "circuit_state": health.circuit_state.value,
                "total_requests": health.total_requests,
                "total_failures": health.total_failures,
                "failure_rate": round(health.failure_rate, 4),
                "consecutive_failures": health.consecutive_failures,
                "last_failure": health.last_failure_time.isoformat() if health.last_failure_time else None,
                "last_success": health.last_success_time.isoformat() if health.last_success_time else None,
            }
        return stats
