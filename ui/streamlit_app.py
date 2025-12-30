"""
RAG Data Assistant - Streamlit Demo UI
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
"""

import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8080")

# Page config
st.set_page_config(
    page_title="RAG Data Assistant",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .sql-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


def query_api(question: str, execute: bool = True) -> dict:
    """Send query to API."""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/query",
            json={"question": question, "execute": execute},
            timeout=60,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def main():
    # Header
    st.markdown('<div class="main-header">RAG Data Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask questions about your data in natural language</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This demo showcases the RAG-powered Data Assistant that converts
        natural language questions into SQL queries.
        
        **Features:**
        - Natural language to SQL
        - RAG-enhanced context
        - Query execution
        - Result explanation
        
        **Example Questions:**
        - Show top 10 customers by revenue
        - What's the average order value?
        - Compare sales this month vs last month
        - Which products are low in stock?
        """)
        
        st.header("Settings")
        execute_query = st.checkbox("Execute SQL", value=True)
        show_sql = st.checkbox("Show Generated SQL", value=True)
        
        st.header("Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries Today", "42")
        with col2:
            st.metric("Avg Latency", "2.3s")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query input
        question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., Show me the top 10 customers by revenue",
        )
        
        # Quick examples
        st.markdown("**Quick Examples:**")
        example_cols = st.columns(4)
        examples = [
            "Top 10 customers by revenue",
            "Average order value",
            "Revenue by country",
            "Low stock products",
        ]
        
        for i, example in enumerate(examples):
            with example_cols[i]:
                if st.button(example, key=f"ex_{i}"):
                    question = example
        
        if question:
            with st.spinner("Processing your query..."):
                result = query_api(question, execute_query)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                # Show generated SQL
                if show_sql and result.get("sql"):
                    st.subheader("Generated SQL")
                    st.code(result["sql"], language="sql")
                
                # Show validation status
                if result.get("is_valid"):
                    st.success("Query validated successfully")
                else:
                    st.warning("Query validation issues detected")
                
                # Show explanation
                if result.get("explanation"):
                    st.subheader("Explanation")
                    st.info(result["explanation"])
                
                # Show results
                if result.get("results") and not result["results"].get("error"):
                    st.subheader("Query Results")
                    
                    results = result["results"]
                    rows = results.get("rows", [])
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True)
                        
                        # Show summary metrics
                        st.markdown(
                            f"**{results.get('row_count', 0)} rows** returned in "
                            f"**{results.get('execution_time_ms', 0):.2f}ms**"
                        )
                        
                        # Visualization option
                        if len(df.columns) >= 2:
                            st.subheader("Visualization")
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            if numeric_cols:
                                chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Pie"])
                                x_col = st.selectbox("X Axis", df.columns.tolist())
                                y_col = st.selectbox("Y Axis", numeric_cols)
                                
                                if chart_type == "Bar":
                                    fig = px.bar(df, x=x_col, y=y_col)
                                elif chart_type == "Line":
                                    fig = px.line(df, x=x_col, y=y_col)
                                else:
                                    fig = px.pie(df, names=x_col, values=y_col)
                                
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No results returned")
                
                # Show sources
                if result.get("sources"):
                    st.subheader("Sources Used")
                    for source in result["sources"]:
                        st.markdown(f"- {source.get('source', 'Unknown')} (score: {source.get('score', 0):.2f})")
    
    with col2:
        # Metrics panel
        st.subheader("Query Metrics")
        
        if question and "error" not in result:
            metrics = result.get("metrics", {})
            st.metric("Latency", f"{metrics.get('latency_ms', 0):.2f} ms")
            st.metric("Tokens Used", metrics.get("tokens_used", 0))
            st.metric("Cost", f"${metrics.get('cost_usd', 0):.4f}")
        
        # Recent queries (placeholder)
        st.subheader("Recent Queries")
        recent = [
            "Top customers by revenue",
            "Monthly sales trend",
            "Product inventory status",
        ]
        for q in recent:
            st.markdown(f"- {q}")


if __name__ == "__main__":
    main()
