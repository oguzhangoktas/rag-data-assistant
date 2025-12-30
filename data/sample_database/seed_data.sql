-- =============================================================================
-- RAG Data Assistant - Sample Seed Data
-- Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
-- =============================================================================

-- Insert sample customers
INSERT INTO customers (email, full_name, registration_date, country, total_lifetime_value) VALUES
('john.doe@email.com', 'John Doe', '2023-01-15', 'US', 2500.00),
('jane.smith@email.com', 'Jane Smith', '2023-02-20', 'UK', 3200.00),
('bob.johnson@email.com', 'Bob Johnson', '2023-03-10', 'CA', 1800.00),
('alice.williams@email.com', 'Alice Williams', '2023-04-05', 'US', 4500.00),
('charlie.brown@email.com', 'Charlie Brown', '2023-05-12', 'DE', 2100.00),
('diana.ross@email.com', 'Diana Ross', '2023-06-18', 'FR', 3800.00),
('edward.jones@email.com', 'Edward Jones', '2023-07-22', 'US', 1500.00),
('fiona.green@email.com', 'Fiona Green', '2023-08-30', 'UK', 2900.00),
('george.white@email.com', 'George White', '2023-09-14', 'US', 5200.00),
('helen.black@email.com', 'Helen Black', '2023-10-08', 'CA', 1200.00);

-- Insert sample products
INSERT INTO products (name, category, price, stock_quantity) VALUES
('Laptop Pro', 'Electronics', 1299.99, 50),
('Wireless Mouse', 'Electronics', 49.99, 200),
('USB-C Hub', 'Electronics', 79.99, 150),
('Mechanical Keyboard', 'Electronics', 149.99, 100),
('Monitor 27inch', 'Electronics', 399.99, 75),
('Office Chair', 'Furniture', 299.99, 40),
('Standing Desk', 'Furniture', 499.99, 30),
('Desk Lamp', 'Furniture', 59.99, 120),
('Notebook Set', 'Office Supplies', 19.99, 500),
('Pen Pack', 'Office Supplies', 9.99, 1000);

-- Insert sample orders
INSERT INTO orders (customer_id, order_date, total_amount, status, payment_method) VALUES
(1, '2024-01-05 10:30:00', 1349.98, 'completed', 'credit_card'),
(2, '2024-01-06 14:20:00', 449.98, 'completed', 'paypal'),
(3, '2024-01-07 09:15:00', 799.98, 'completed', 'credit_card'),
(4, '2024-01-08 16:45:00', 1899.97, 'completed', 'credit_card'),
(5, '2024-01-09 11:00:00', 149.99, 'completed', 'bank_transfer'),
(1, '2024-01-10 13:30:00', 129.98, 'completed', 'credit_card'),
(6, '2024-01-11 15:20:00', 559.98, 'pending', 'paypal'),
(7, '2024-01-12 10:00:00', 49.99, 'completed', 'credit_card'),
(8, '2024-01-13 14:45:00', 2099.96, 'completed', 'credit_card'),
(9, '2024-01-14 09:30:00', 899.97, 'cancelled', 'paypal');

-- Insert sample order items
INSERT INTO order_items (order_id, product_id, quantity, price) VALUES
(1, 1, 1, 1299.99),
(1, 2, 1, 49.99),
(2, 5, 1, 399.99),
(2, 2, 1, 49.99),
(3, 6, 1, 299.99),
(3, 7, 1, 499.99),
(4, 1, 1, 1299.99),
(4, 3, 2, 79.99),
(4, 4, 1, 149.99),
(5, 4, 1, 149.99),
(6, 3, 1, 79.99),
(6, 2, 1, 49.99),
(7, 8, 2, 59.99),
(7, 5, 1, 399.99),
(8, 2, 1, 49.99),
(9, 1, 1, 1299.99),
(9, 5, 1, 399.99),
(10, 6, 3, 299.99);

-- Update customer lifetime values based on orders
UPDATE customers c SET total_lifetime_value = (
    SELECT COALESCE(SUM(o.total_amount), 0)
    FROM orders o
    WHERE o.customer_id = c.customer_id
    AND o.status = 'completed'
);
