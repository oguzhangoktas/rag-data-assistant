"""
Database Schema Manager
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Manages database schema information for SQL generation,
including table descriptions, columns, and relationships.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("schema_manager")


@dataclass
class Column:
    """Represents a database column."""
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[str] = None
    description: Optional[str] = None
    is_pii: bool = False


@dataclass
class Table:
    """Represents a database table."""
    name: str
    description: str
    columns: List[Column]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    sample_queries: List[Dict[str, str]]
    is_system_table: bool = False


class SchemaManager:
    """
    Manages database schema for SQL generation.
    
    Usage:
        manager = SchemaManager()
        prompt = manager.get_schema_prompt()
    """
    
    def __init__(self):
        """Initialize the schema manager."""
        self._config = get_config()
        self._schema_config = self._config.get_schema_config()
        self._tables: Dict[str, Table] = {}
        self._load_schema()
        logger.info(f"SchemaManager initialized with {len(self._tables)} tables")
    
    def _load_schema(self) -> None:
        """Load schema from configuration."""
        schemas = self._schema_config.get("schemas", {})
        
        for schema_name, schema_def in schemas.items():
            tables = schema_def.get("tables", {})
            
            for table_name, table_def in tables.items():
                columns = []
                
                for col_name, col_def in table_def.get("columns", {}).items():
                    columns.append(Column(
                        name=col_name,
                        data_type=col_def.get("type", "VARCHAR"),
                        nullable=col_def.get("nullable", True),
                        primary_key=col_def.get("primary_key", False),
                        foreign_key=col_def.get("foreign_key"),
                        description=col_def.get("description", ""),
                        is_pii=col_def.get("pii", False),
                    ))
                
                primary_keys = [c.name for c in columns if c.primary_key]
                
                foreign_keys = []
                for rel in table_def.get("relationships", []):
                    foreign_keys.append({
                        "table": rel.get("table"),
                        "columns": rel.get("columns", []),
                        "type": rel.get("type", "many_to_one"),
                    })
                
                self._tables[table_name] = Table(
                    name=table_name,
                    description=table_def.get("description", ""),
                    columns=columns,
                    primary_keys=primary_keys,
                    foreign_keys=foreign_keys,
                    sample_queries=table_def.get("sample_queries", []),
                    is_system_table=table_def.get("is_system_table", False),
                )
    
    def get_table(self, table_name: str) -> Optional[Table]:
        """Get a table by name."""
        return self._tables.get(table_name)
    
    def get_all_tables(self) -> List[Table]:
        """Get all tables (excluding system tables)."""
        return [t for t in self._tables.values() if not t.is_system_table]
    
    def get_schema_prompt(self, include_samples: bool = True) -> str:
        """Generate a schema description prompt for LLM."""
        lines = ["DATABASE SCHEMA:", ""]
        
        for table in self.get_all_tables():
            lines.append(f"Table: {table.name}")
            lines.append(f"Description: {table.description}")
            lines.append("Columns:")
            
            for col in table.columns:
                col_info = f"  - {col.name} ({col.data_type})"
                if col.primary_key:
                    col_info += " [PK]"
                if col.foreign_key:
                    col_info += f" [FK -> {col.foreign_key}]"
                if col.description:
                    col_info += f": {col.description}"
                lines.append(col_info)
            
            if include_samples and table.sample_queries:
                lines.append("Example Queries:")
                for sq in table.sample_queries[:2]:
                    lines.append(f"  Q: {sq.get('question', '')}")
                    lines.append(f"  SQL: {sq.get('sql', '')}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def find_relevant_tables(self, query: str) -> List[str]:
        """Find tables relevant to a natural language query."""
        query_lower = query.lower()
        relevant = set()
        
        for table in self._tables.values():
            if table.name.lower() in query_lower:
                relevant.add(table.name)
                continue
            
            for col in table.columns:
                if col.name.lower().replace("_", " ") in query_lower:
                    relevant.add(table.name)
                    break
        
        return list(relevant)


_schema_manager: Optional[SchemaManager] = None


def get_schema_manager() -> SchemaManager:
    """Get the global schema manager instance."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = SchemaManager()
    return _schema_manager
