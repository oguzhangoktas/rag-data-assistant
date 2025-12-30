-- =============================================================================
-- RAG Data Assistant - Sample Database Schema
-- Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
-- =============================================================================

-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    full_name VARCHAR(255) NOT NULL,
    registration_date DATE NOT NULL DEFAULT CURRENT_DATE,
    country VARCHAR(100),
    total_lifetime_value DECIMAL(15,2) DEFAULT 0
);

CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_registration ON customers(registration_date);
CREATE INDEX idx_customers_country ON customers(country);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    price DECIMAL(15,2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0
);

CREATE INDEX idx_products_category ON products(category);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TIMESTAMP NOT NULL DEFAULT NOW(),
    total_amount DECIMAL(15,2) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    payment_method VARCHAR(50)
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(status);

-- Order items table
CREATE TABLE IF NOT EXISTS order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    price DECIMAL(15,2) NOT NULL
);

CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

-- Query history table (system)
CREATE TABLE IF NOT EXISTS query_history (
    query_id SERIAL PRIMARY KEY,
    user_question TEXT NOT NULL,
    generated_sql TEXT NOT NULL,
    execution_time_ms INTEGER,
    row_count INTEGER,
    feedback_score INTEGER CHECK (feedback_score BETWEEN 1 AND 5),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_query_history_created ON query_history(created_at);
