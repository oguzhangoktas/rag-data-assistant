"""
Configuration Loader - Loads and validates YAML configuration files
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMSettings(BaseModel):
    """LLM provider settings."""
    primary_provider: str = Field(default="openai")
    fallback_provider: str = Field(default="anthropic")
    model_name: str = Field(default="gpt-4-turbo-preview")
    fallback_model: str = Field(default="claude-3-sonnet-20240229")
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=2048, gt=0)
    timeout_seconds: int = Field(default=30, gt=0)


class EmbeddingSettings(BaseModel):
    """Embedding model settings."""
    model: str = Field(default="text-embedding-3-small")
    dimensions: int = Field(default=1536, gt=0)
    batch_size: int = Field(default=100, gt=0)


class RAGSettings(BaseModel):
    """RAG pipeline settings."""
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k_results: int = Field(default=5, gt=0)
    similarity_threshold: float = Field(default=0.7, ge=0, le=1)
    rerank_enabled: bool = Field(default=True)


class DatabaseSettings(BaseModel):
    """Database connection settings."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="dataassistant")
    user: str = Field(default="assistant")
    password: str = Field(default="")
    
    @property
    def url(self) -> str:
        """Generate database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseModel):
    """Redis cache settings."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    password: str = Field(default="")
    db: int = Field(default=0)
    
    @property
    def url(self) -> str:
        """Generate Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class APISettings(BaseModel):
    """API server settings."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    debug: bool = Field(default=False)
    workers: int = Field(default=4)
    rate_limit_requests: int = Field(default=100)
    rate_limit_period: int = Field(default=3600)


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    
    # LLM Settings
    llm_primary_provider: str = Field(default="openai", alias="LLM_PRIMARY_PROVIDER")
    llm_model_name: str = Field(default="gpt-4-turbo-preview", alias="LLM_MODEL_NAME")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    
    # Database
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="dataassistant", alias="POSTGRES_DB")
    postgres_user: str = Field(default="assistant", alias="POSTGRES_USER")
    postgres_password: str = Field(default="", alias="POSTGRES_PASSWORD")
    
    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    
    # ChromaDB
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection: str = Field(default="documentation", alias="CHROMA_COLLECTION_NAME")
    
    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT")
    api_debug: bool = Field(default=False, alias="API_DEBUG")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def database_url(self) -> str:
        """Generate PostgreSQL database URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def async_database_url(self) -> str:
        """Generate async PostgreSQL database URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


class ConfigLoader:
    """
    Loads and manages application configuration from YAML files and environment variables.
    
    Usage:
        config = ConfigLoader()
        llm_config = config.get_llm_config()
        rag_config = config.get_rag_config()
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Path to configuration directory. Defaults to project config/
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to config/ directory relative to project root
            self.config_dir = Path(__file__).parent.parent.parent / "config"
        
        self._settings = Settings()
        self._llm_config: Optional[Dict[str, Any]] = None
        self._rag_config: Optional[Dict[str, Any]] = None
        self._schema_config: Optional[Dict[str, Any]] = None
        self._prompts_config: Optional[Dict[str, Any]] = None
    
    @property
    def settings(self) -> Settings:
        """Get environment-based settings."""
        return self._settings
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            filename: Name of the YAML file to load
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML is invalid
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    @lru_cache(maxsize=1)
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Load LLM configuration from llm_config.yaml.
        
        Returns:
            LLM configuration dictionary
        """
        if self._llm_config is None:
            self._llm_config = self._load_yaml("llm_config.yaml")
        return self._llm_config
    
    @lru_cache(maxsize=1)
    def get_rag_config(self) -> Dict[str, Any]:
        """
        Load RAG configuration from rag_config.yaml.
        
        Returns:
            RAG configuration dictionary
        """
        if self._rag_config is None:
            self._rag_config = self._load_yaml("rag_config.yaml")
        return self._rag_config
    
    @lru_cache(maxsize=1)
    def get_schema_config(self) -> Dict[str, Any]:
        """
        Load database schema configuration from database_schema.yaml.
        
        Returns:
            Database schema configuration dictionary
        """
        if self._schema_config is None:
            self._schema_config = self._load_yaml("database_schema.yaml")
        return self._schema_config
    
    @lru_cache(maxsize=1)
    def get_prompts_config(self) -> Dict[str, Any]:
        """
        Load prompt templates from prompts.yaml.
        
        Returns:
            Prompts configuration dictionary
        """
        if self._prompts_config is None:
            self._prompts_config = self._load_yaml("prompts.yaml")
        return self._prompts_config
    
    def get_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """
        Get a specific prompt template by name.
        
        Args:
            prompt_name: Name of the prompt template
            
        Returns:
            Prompt template dictionary
            
        Raises:
            KeyError: If the prompt doesn't exist
        """
        prompts = self.get_prompts_config()
        if prompt_name not in prompts.get("prompts", {}):
            raise KeyError(f"Prompt template not found: {prompt_name}")
        return prompts["prompts"][prompt_name]
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema definition for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table schema dictionary or None if not found
        """
        schema = self.get_schema_config()
        for schema_name, schema_def in schema.get("schemas", {}).items():
            if table_name in schema_def.get("tables", {}):
                return schema_def["tables"][table_name]
        return None
    
    def get_all_tables(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all table schemas.
        
        Returns:
            Dictionary of all table schemas
        """
        schema = self.get_schema_config()
        all_tables = {}
        for schema_name, schema_def in schema.get("schemas", {}).items():
            for table_name, table_def in schema_def.get("tables", {}).items():
                all_tables[table_name] = table_def
        return all_tables
    
    def validate(self) -> bool:
        """
        Validate all configuration files exist and are valid.
        
        Returns:
            True if all configurations are valid
            
        Raises:
            ConfigurationException: If validation fails
        """
        required_files = [
            "llm_config.yaml",
            "rag_config.yaml",
            "database_schema.yaml",
            "prompts.yaml",
        ]
        
        for filename in required_files:
            try:
                self._load_yaml(filename)
            except Exception as e:
                from src.utils.exceptions import ConfigurationException
                raise ConfigurationException(f"Invalid configuration {filename}: {e}")
        
        return True


# Global singleton instance
@lru_cache(maxsize=1)
def get_config() -> ConfigLoader:
    """
    Get the global configuration loader instance.
    
    Returns:
        ConfigLoader singleton instance
    """
    return ConfigLoader()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings singleton instance
    """
    return Settings()
