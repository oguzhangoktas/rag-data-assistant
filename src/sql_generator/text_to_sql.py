"""
Text-to-SQL Generator
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Converts natural language queries to SQL using LLM with RAG context
and schema awareness.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.llm.llm_client import LLMClient, get_llm_client, LLMResponse
from src.llm.prompt_manager import PromptManager, get_prompt_manager
from src.llm.response_validator import ResponseValidator
from src.rag.retriever import Retriever, get_retriever
from src.sql_generator.schema_manager import SchemaManager, get_schema_manager
from src.sql_generator.query_validator import QueryValidator
from src.utils.logger import get_logger
from src.utils.exceptions import SQLGenerationException

logger = get_logger("text_to_sql")


@dataclass
class SQLGenerationResult:
    """Result from SQL generation."""
    sql: str
    is_valid: bool
    explanation: str
    confidence: float
    rag_context: str
    sources: List[Dict[str, str]]
    latency_ms: float
    tokens_used: int
    cost_usd: float
    validation_issues: List[str]


class TextToSQL:
    """
    Converts natural language questions to SQL queries.
    
    Features:
    - Schema-aware SQL generation
    - RAG-enhanced context from documentation
    - Few-shot learning from query history
    - SQL validation and security checks
    
    Usage:
        generator = TextToSQL()
        result = await generator.generate("Show top 10 customers by revenue")
        if result.is_valid:
            print(result.sql)
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        retriever: Optional[Retriever] = None,
        schema_manager: Optional[SchemaManager] = None,
    ):
        """
        Initialize the text-to-SQL generator.
        
        Args:
            llm_client: LLMClient instance
            prompt_manager: PromptManager instance
            retriever: Retriever instance for RAG
            schema_manager: SchemaManager instance
        """
        self._llm = llm_client or get_llm_client()
        self._prompts = prompt_manager or get_prompt_manager()
        self._retriever = retriever or get_retriever()
        self._schema = schema_manager or get_schema_manager()
        self._validator = QueryValidator()
        self._response_validator = ResponseValidator()
        
        # Few-shot examples storage
        self._few_shot_examples: List[Dict[str, str]] = []
        self._max_few_shot = 5
        
        logger.info("TextToSQL initialized")
    
    async def generate(
        self,
        question: str,
        use_rag: bool = True,
        use_few_shot: bool = True,
        validate: bool = True,
    ) -> SQLGenerationResult:
        """
        Generate SQL from a natural language question.
        
        Args:
            question: Natural language question
            use_rag: Whether to use RAG for context
            use_few_shot: Whether to include few-shot examples
            validate: Whether to validate generated SQL
            
        Returns:
            SQLGenerationResult with SQL and metadata
        """
        start_time = time.time()
        
        # Get schema context
        schema_context = self._schema.get_schema_prompt()
        
        # Get RAG context if enabled
        rag_context = ""
        sources = []
        if use_rag:
            try:
                retrieval_result = await self._retriever.retrieve(question)
                rag_context = retrieval_result.context
                sources = retrieval_result.sources
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Build few-shot examples
        few_shot_str = ""
        if use_few_shot and self._few_shot_examples:
            examples = self._few_shot_examples[-self._max_few_shot:]
            few_shot_str = "\n\n".join([
                f"Question: {ex['question']}\nSQL: {ex['sql']}"
                for ex in examples
            ])
        
        # Build the prompt
        prompt = self._prompts.build_text_to_sql_prompt(
            user_question=question,
            schema_context=schema_context,
            rag_context=rag_context,
            few_shot_examples=few_shot_str,
        )
        
        # Generate SQL using LLM
        try:
            system_message = self._prompts.get_system_message("sql_mode")
            
            response = await self._llm.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1,  # Low temperature for SQL
                max_tokens=500,
            )
            
            # Extract and validate SQL
            validation = self._response_validator.validate_sql(response.content)
            sql = validation.cleaned_response
            
            validation_issues = validation.issues + validation.warnings
            is_valid = validation.is_valid
            
            # Additional query validation
            if validate and is_valid:
                query_validation = self._validator.validate(sql, self._schema)
                if not query_validation.is_valid:
                    is_valid = False
                    validation_issues.extend(query_validation.issues)
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = SQLGenerationResult(
                sql=sql,
                is_valid=is_valid,
                explanation="",  # Will be generated separately if needed
                confidence=0.9 if is_valid else 0.5,
                rag_context=rag_context,
                sources=sources,
                latency_ms=latency_ms,
                tokens_used=response.total_tokens,
                cost_usd=response.cost_usd,
                validation_issues=validation_issues,
            )
            
            # Store as few-shot example if valid
            if is_valid:
                self.add_few_shot_example(question, sql)
            
            return result
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise SQLGenerationException(
                message=f"Failed to generate SQL: {e}",
                query=question,
            )
    
    async def generate_with_explanation(
        self,
        question: str,
    ) -> SQLGenerationResult:
        """
        Generate SQL with a natural language explanation.
        
        Args:
            question: Natural language question
            
        Returns:
            SQLGenerationResult with SQL and explanation
        """
        # First generate the SQL
        result = await self.generate(question)
        
        if not result.is_valid:
            result.explanation = f"Could not generate valid SQL: {', '.join(result.validation_issues)}"
            return result
        
        # Generate explanation
        try:
            explanation_prompt = self._prompts.format_prompt(
                "sql_explanation",
                user_question=question,
                sql_query=result.sql,
                query_results="[Not yet executed]",
                rag_context=result.rag_context,
            )
            
            explanation_response = await self._llm.generate(
                prompt=explanation_prompt,
                system_message=self._prompts.get_system_message("analyst_mode"),
                temperature=0.3,
                max_tokens=300,
            )
            
            result.explanation = explanation_response.content
            result.tokens_used += explanation_response.total_tokens
            result.cost_usd += explanation_response.cost_usd
            
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            result.explanation = "SQL generated successfully."
        
        return result
    
    def add_few_shot_example(
        self,
        question: str,
        sql: str,
    ) -> None:
        """
        Add a successful query as a few-shot example.
        
        Args:
            question: Natural language question
            sql: Corresponding SQL query
        """
        example = {"question": question, "sql": sql}
        
        # Avoid duplicates
        for ex in self._few_shot_examples:
            if ex["question"].lower() == question.lower():
                return
        
        self._few_shot_examples.append(example)
        
        # Keep only recent examples
        if len(self._few_shot_examples) > self._max_few_shot * 2:
            self._few_shot_examples = self._few_shot_examples[-self._max_few_shot:]
        
        logger.debug(f"Added few-shot example, total: {len(self._few_shot_examples)}")
    
    def clear_few_shot_examples(self) -> None:
        """Clear all few-shot examples."""
        self._few_shot_examples.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "few_shot_examples": len(self._few_shot_examples),
            "max_few_shot": self._max_few_shot,
        }


# Global singleton instance
_text_to_sql: Optional[TextToSQL] = None


def get_text_to_sql() -> TextToSQL:
    """Get the global TextToSQL instance."""
    global _text_to_sql
    if _text_to_sql is None:
        _text_to_sql = TextToSQL()
    return _text_to_sql
