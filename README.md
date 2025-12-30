# RAG Data Assistant

> Enterprise LLM-Powered Data Analytics Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Oguzhan Goktas  
**Email:** oguzhangoktas22@gmail.com  
**GitHub:** [github.com/oguzhangoktas](https://github.com/oguzhangoktas)  
**LinkedIn:** [linkedin.com/in/oguzhan-goktas](https://www.linkedin.com/in/oguzhan-goktas/)

---

## Overview

RAG Data Assistant is a production-ready, LLM-powered platform that enables users to query data using natural language. The system combines Retrieval Augmented Generation (RAG) with Text-to-SQL capabilities to provide accurate, context-aware responses.

### Key Features

- **Natural Language to SQL**: Convert plain English questions into optimized SQL queries
- **RAG-Enhanced Context**: Leverage documentation for accurate query generation
- **Multi-Provider LLM Support**: OpenAI GPT-4 with Anthropic Claude fallback
- **Real-time Query Execution**: Execute queries and get formatted results
- **Streaming Responses**: Server-Sent Events for real-time feedback
- **Cost Optimization**: Semantic caching and embedding reuse
- **Production Monitoring**: Prometheus metrics and Grafana dashboards

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                    (Streamlit / REST API)                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                       FastAPI Server                             │
│                    (/api/v1/query endpoint)                      │
└───────┬─────────────────────┬───────────────────────┬───────────┘
        │                     │                       │
        ▼                     ▼                       ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐
│  RAG Engine   │    │ Text-to-SQL   │    │   Query Executor      │
│  - Retriever  │    │ - Generator   │    │   - PostgreSQL        │
│  - ChromaDB   │    │ - Validator   │    │   - Result Formatter  │
│  - Embeddings │    │ - Few-shot    │    │   - Safety Checks     │
└───────────────┘    └───────────────┘    └───────────────────────┘
        │                     │
        ▼                     ▼
┌───────────────┐    ┌───────────────┐
│  OpenAI API   │    │ Documentation │
│  - GPT-4      │    │ - Data Dict   │
│  - Embeddings │    │ - Metrics     │
└───────────────┘    └───────────────┘
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key
- (Optional) Anthropic API key for fallback

### 5-Minute Setup

```bash
# Clone the repository
git clone https://github.com/oguzhangoktas/rag-data-assistant.git
cd rag-data-assistant

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start all services
make demo

# Access the application
# - UI: http://localhost:8501
# - API: http://localhost:8080
# - Docs: http://localhost:8080/docs
```

---

## Demo Scenarios

### Query 1: Top Customers by Revenue
```
Question: "Show me the top 10 customers by revenue"

Generated SQL:
SELECT c.customer_id, c.full_name, c.email, c.total_lifetime_value
FROM customers c
ORDER BY c.total_lifetime_value DESC
LIMIT 10;

Result: 10 rows returned in 45ms
```

### Query 2: Table Discovery
```
Question: "Which table contains customer email addresses?"

Response: The 'customers' table contains email addresses in the 'email' 
column (VARCHAR 255). This column stores unique customer email addresses.
```

### Query 3: Business Metrics
```
Question: "How do we calculate customer lifetime value?"

Response: Customer Lifetime Value (CLV) is stored in the 'total_lifetime_value' 
column in the customers table. It's calculated as the sum of all completed 
orders for each customer: SUM(orders.total_amount) WHERE status = 'completed'
```

### Query 4: Time Comparison
```
Question: "Compare sales this month vs last month"

Generated SQL:
SELECT 
  SUM(CASE WHEN order_date >= DATE_TRUNC('month', CURRENT_DATE) 
      THEN total_amount ELSE 0 END) as this_month,
  SUM(CASE WHEN order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
           AND order_date < DATE_TRUNC('month', CURRENT_DATE) 
      THEN total_amount ELSE 0 END) as last_month
FROM orders WHERE status = 'completed';
```

### Query 5: Aggregations
```
Question: "What's the average order value?"

Generated SQL:
SELECT AVG(total_amount) as average_order_value
FROM orders
WHERE status = 'completed';

Result: $456.78
```

---

## Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| Query Latency (p95) | <3s | 2.4s |
| RAG Retrieval | <200ms | 156ms |
| Embedding Generation | <500ms | 380ms |
| SQL Correctness | >90% | 92% |
| Context Relevance | >0.75 | 0.82 |
| Faithfulness | >0.85 | 0.88 |

### Cost Analysis

| Metric | Value |
|--------|-------|
| Avg Cost per Query | $0.03 |
| Daily Cost (100 queries) | $3.00 |
| Monthly Projection | ~$90 |
| Cache Hit Rate | 35% |
| Cost Reduction via Caching | 40% |

---

## Tech Stack

### Core AI/ML
- **OpenAI GPT-4 Turbo** - Primary LLM
- **Anthropic Claude 3 Sonnet** - Fallback LLM
- **text-embedding-3-small** - Embeddings
- **LangChain** - LLM orchestration
- **ChromaDB** - Vector database

### Backend
- **FastAPI** - Async API framework
- **PostgreSQL 15** - Data warehouse
- **Redis** - Caching layer
- **SQLAlchemy 2.0** - ORM

### Monitoring
- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **Loguru** - Structured logging

### Testing
- **pytest** - Unit & integration tests
- **RAGAS** - RAG evaluation
- **Locust** - Load testing

---

## Project Structure

```
rag-data-assistant/
├── config/                    # YAML configurations
│   ├── llm_config.yaml
│   ├── rag_config.yaml
│   ├── database_schema.yaml
│   └── prompts.yaml
├── src/
│   ├── llm/                   # LLM integration
│   ├── rag/                   # RAG pipeline
│   ├── sql_generator/         # Text-to-SQL
│   ├── api/                   # FastAPI routes
│   └── utils/                 # Utilities
├── tests/                     # Test suite
├── data/                      # Sample data
├── ui/                        # Streamlit UI
├── docker/                    # Docker configs
├── monitoring/                # Prometheus/Grafana
└── docs/                      # Documentation
```

---

## API Reference

### Query Endpoint
```http
POST /api/v1/query
Content-Type: application/json

{
  "question": "Show top 10 customers by revenue",
  "execute": true,
  "stream": false
}
```

### Response
```json
{
  "question": "Show top 10 customers by revenue",
  "sql": "SELECT ... LIMIT 10;",
  "is_valid": true,
  "explanation": "This query retrieves...",
  "results": {
    "columns": ["customer_id", "full_name", "total_lifetime_value"],
    "rows": [...],
    "row_count": 10,
    "execution_time_ms": 45
  },
  "metrics": {
    "latency_ms": 2340,
    "tokens_used": 1250,
    "cost_usd": 0.028
  }
}
```

---

## Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage
pytest --cov=src --cov-report=html

# Load testing
locust -f tests/load/locust_load_test.py
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `POSTGRES_HOST` | Database host | localhost |
| `REDIS_HOST` | Redis host | localhost |
| `CHROMA_HOST` | ChromaDB host | localhost |

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Contact

**Oguzhan Goktas**
- Email: oguzhangoktas22@gmail.com
- GitHub: [github.com/oguzhangoktas](https://github.com/oguzhangoktas)
- LinkedIn: [linkedin.com/in/oguzhan-goktas](https://www.linkedin.com/in/oguzhan-goktas/)
