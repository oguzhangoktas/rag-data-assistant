# Common Query Examples

## Customer Queries

### Get Top 10 Customers by Revenue
```sql
SELECT 
    c.customer_id,
    c.full_name,
    c.email,
    c.total_lifetime_value
FROM customers c
ORDER BY c.total_lifetime_value DESC
LIMIT 10;
```

### Find Customer by Email
```sql
SELECT * FROM customers WHERE email = 'john.doe@email.com';
```

### Customers by Country
```sql
SELECT 
    country,
    COUNT(*) as customer_count,
    SUM(total_lifetime_value) as total_revenue
FROM customers
GROUP BY country
ORDER BY total_revenue DESC;
```

## Order Queries

### Orders in Date Range
```sql
SELECT 
    o.order_id,
    c.full_name,
    o.order_date,
    o.total_amount,
    o.status
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date BETWEEN '2024-01-01' AND '2024-01-31'
ORDER BY o.order_date DESC;
```

### Revenue by Status
```sql
SELECT 
    status,
    COUNT(*) as order_count,
    SUM(total_amount) as revenue
FROM orders
GROUP BY status;
```

### Daily Revenue Trend
```sql
SELECT 
    DATE(order_date) as order_day,
    COUNT(*) as orders,
    SUM(total_amount) as daily_revenue
FROM orders
WHERE status = 'completed'
GROUP BY DATE(order_date)
ORDER BY order_day DESC
LIMIT 30;
```

## Product Queries

### Products by Category
```sql
SELECT 
    category,
    COUNT(*) as product_count,
    AVG(price) as avg_price
FROM products
GROUP BY category;
```

### Low Stock Products
```sql
SELECT 
    product_id,
    name,
    category,
    stock_quantity
FROM products
WHERE stock_quantity < 10
ORDER BY stock_quantity ASC;
```

### Best Selling Products
```sql
SELECT 
    p.product_id,
    p.name,
    p.category,
    SUM(oi.quantity) as total_sold,
    SUM(oi.quantity * oi.price) as revenue
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'completed'
GROUP BY p.product_id, p.name, p.category
ORDER BY total_sold DESC
LIMIT 10;
```

## Comparison Queries

### This Month vs Last Month Revenue
```sql
SELECT 
    SUM(CASE WHEN order_date >= DATE_TRUNC('month', CURRENT_DATE) 
        THEN total_amount ELSE 0 END) as this_month_revenue,
    SUM(CASE WHEN order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
             AND order_date < DATE_TRUNC('month', CURRENT_DATE)
        THEN total_amount ELSE 0 END) as last_month_revenue
FROM orders
WHERE status = 'completed';
```

### Revenue by Region
```sql
SELECT 
    c.country,
    SUM(o.total_amount) as revenue,
    COUNT(DISTINCT o.order_id) as orders
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.country
ORDER BY revenue DESC;
```

## Average Order Value
```sql
SELECT 
    AVG(total_amount) as avg_order_value,
    MIN(total_amount) as min_order,
    MAX(total_amount) as max_order
FROM orders
WHERE status = 'completed';
```
