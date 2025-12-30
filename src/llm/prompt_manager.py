"""
Prompt Template Manager for LLM Interactions
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Manages loading, formatting, and versioning of prompt templates for various
LLM tasks including SQL generation, documentation search, and result explanation.
"""

import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from functools import lru_cache

from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.utils.exceptions import ConfigurationException

logger = get_logger("prompt_manager")


@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata."""
    name: str
    version: str
    description: str
    template: str
    input_variables: List[str]
    output_parser: str
    max_tokens: int


class PromptManager:
    """
    Manages prompt templates for LLM interactions.
    
    Features:
    - Load templates from YAML configuration
    - Variable substitution with validation
    - Template versioning
    - Caching for performance
    
    Usage:
        manager = PromptManager()
        prompt = manager.format_prompt(
            "text_to_sql",
            schema_context="...",
            user_question="Show top customers"
        )
    """
    
    def __init__(self):
        """Initialize the prompt manager."""
        self._config = get_config()
        self._prompts_config = self._config.get_prompts_config()
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_templates()
        
        logger.info(f"PromptManager initialized with {len(self._templates)} templates")
    
    def _load_templates(self) -> None:
        """Load all prompt templates from configuration."""
        prompts = self._prompts_config.get("prompts", {})
        
        for name, prompt_data in prompts.items():
            try:
                self._templates[name] = PromptTemplate(
                    name=name,
                    version=prompt_data.get("version", "1.0"),
                    description=prompt_data.get("description", ""),
                    template=prompt_data.get("template", ""),
                    input_variables=prompt_data.get("input_variables", []),
                    output_parser=prompt_data.get("output_parser", "text"),
                    max_tokens=prompt_data.get("max_tokens", 500),
                )
            except Exception as e:
                logger.warning(f"Failed to load prompt template '{name}': {e}")
    
    def get_template(self, name: str) -> PromptTemplate:
        """
        Get a prompt template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate object
            
        Raises:
            ConfigurationException: If template not found
        """
        if name not in self._templates:
            raise ConfigurationException(
                message=f"Prompt template '{name}' not found",
                config_key=f"prompts.{name}",
            )
        return self._templates[name]
    
    def format_prompt(self, name: str, **kwargs) -> str:
        """
        Format a prompt template with provided variables.
        
        Args:
            name: Template name
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
            
        Raises:
            ConfigurationException: If template not found
            ValueError: If required variables are missing
        """
        template = self.get_template(name)
        
        # Check for missing required variables
        missing_vars = []
        for var in template.input_variables:
            if var not in kwargs:
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(
                f"Missing required variables for prompt '{name}': {missing_vars}"
            )
        
        # Format the template
        try:
            formatted = template.template.format(**kwargs)
            return formatted.strip()
        except KeyError as e:
            raise ValueError(f"Missing variable in template: {e}")
    
    def get_system_message(self, mode: str = "default") -> str:
        """
        Get a system message by mode.
        
        Args:
            mode: System message mode (default, sql_mode, analyst_mode)
            
        Returns:
            System message string
        """
        system_messages = self._prompts_config.get("system_messages", {})
        return system_messages.get(mode, system_messages.get("default", ""))
    
    def get_error_message(self, error_type: str) -> str:
        """
        Get an error message template.
        
        Args:
            error_type: Type of error (no_results, query_timeout, etc.)
            
        Returns:
            Error message string
        """
        error_messages = self._prompts_config.get("error_messages", {})
        return error_messages.get(error_type, "An error occurred. Please try again.")
    
    def build_text_to_sql_prompt(
        self,
        user_question: str,
        schema_context: str,
        rag_context: str = "",
        few_shot_examples: str = "",
    ) -> str:
        """
        Build a complete text-to-SQL prompt.
        
        Args:
            user_question: User's natural language question
            schema_context: Database schema description
            rag_context: Retrieved documentation context
            few_shot_examples: Example queries for few-shot learning
            
        Returns:
            Complete formatted prompt
        """
        return self.format_prompt(
            "text_to_sql",
            user_question=user_question,
            schema_context=schema_context,
            rag_context=rag_context or "No additional documentation available.",
            few_shot_examples=few_shot_examples or "No examples available.",
        )
    
    def build_explanation_prompt(
        self,
        user_question: str,
        sql_query: str,
        query_results: str,
        rag_context: str = "",
    ) -> str:
        """
        Build a prompt for explaining SQL query results.
        
        Args:
            user_question: Original user question
            sql_query: Executed SQL query
            query_results: Query result data
            rag_context: Retrieved documentation context
            
        Returns:
            Complete formatted prompt
        """
        return self.format_prompt(
            "sql_explanation",
            user_question=user_question,
            sql_query=sql_query,
            query_results=query_results,
            rag_context=rag_context or "No additional context.",
        )
    
    def build_documentation_search_prompt(
        self,
        user_question: str,
        rag_context: str,
    ) -> str:
        """
        Build a prompt for answering documentation questions.
        
        Args:
            user_question: User's question
            rag_context: Retrieved documentation chunks
            
        Returns:
            Complete formatted prompt
        """
        return self.format_prompt(
            "documentation_search",
            user_question=user_question,
            rag_context=rag_context,
        )
    
    def build_query_refinement_prompt(
        self,
        user_question: str,
        schema_context: str,
    ) -> str:
        """
        Build a prompt for refining ambiguous queries.
        
        Args:
            user_question: User's ambiguous question
            schema_context: Database schema for context
            
        Returns:
            Complete formatted prompt
        """
        return self.format_prompt(
            "query_refinement",
            user_question=user_question,
            schema_context=schema_context,
        )
    
    def build_validation_prompt(
        self,
        sql_query: str,
        schema_context: str,
    ) -> str:
        """
        Build a prompt for SQL validation.
        
        Args:
            sql_query: SQL query to validate
            schema_context: Database schema for validation
            
        Returns:
            Complete formatted prompt
        """
        return self.format_prompt(
            "sql_validation",
            sql_query=sql_query,
            schema_context=schema_context,
        )
    
    def build_hallucination_check_prompt(
        self,
        rag_context: str,
        query_results: str,
        response: str,
    ) -> str:
        """
        Build a prompt for hallucination detection.
        
        Args:
            rag_context: Source documentation context
            query_results: Database query results
            response: LLM response to check
            
        Returns:
            Complete formatted prompt
        """
        return self.format_prompt(
            "hallucination_check",
            rag_context=rag_context,
            query_results=query_results,
            response=response,
        )
    
    def list_templates(self) -> List[str]:
        """
        List all available template names.
        
        Returns:
            List of template names
        """
        return list(self._templates.keys())
    
    def get_template_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a template.
        
        Args:
            name: Template name
            
        Returns:
            Dictionary with template metadata
        """
        template = self.get_template(name)
        return {
            "name": template.name,
            "version": template.version,
            "description": template.description,
            "input_variables": template.input_variables,
            "output_parser": template.output_parser,
            "max_tokens": template.max_tokens,
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4


# Global singleton instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """
    Get the global prompt manager instance.
    
    Returns:
        PromptManager singleton instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
