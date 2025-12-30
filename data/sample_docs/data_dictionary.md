# Data Dictionary

## Overview
This document describes the data model for the e-commerce analytics platform.

## Tables

### customers
Contains customer master data including demographics and purchase history.

| Column | Type | Description |
|--------|------|-------------|
| customer_id | INTEGER | Unique customer identifier (Primary Key) |
| email | VARCHAR(255) | Customer email address (unique) |
| full_name | VARCHAR(255) | Customer full name |
| registration_date | DATE | Date when customer registered |
| country | VARCHAR(100) | Customer country (ISO code) |
| total_lifetime_value | DECIMAL(15,2) | Total revenue from this customer |

### orders
Contains order transactions with status and payment information.

| Column | Type | Description |
|--------|------|-------------|
| order_id | INTEGER | Unique order identifier (Primary Key) |
| customer_id | INTEGER | Reference to customer (Foreign Key) |
| order_date | TIMESTAMP | Date and time of order |
| total_amount | DECIMAL(15,2) | Total order amount in USD |
| status | VARCHAR(50) | Order status: pending, completed, cancelled, refunded |
| payment_method | VARCHAR(50) | Payment method used |

### products
Contains product catalog information.

| Column | Type | Description |
|--------|------|-------------|
| product_id | INTEGER | Unique product identifier (Primary Key) |
| name | VARCHAR(255) | Product name |
| category | VARCHAR(100) | Product category |
| price | DECIMAL(15,2) | Current price in USD |
| stock_quantity | INTEGER | Current inventory count |

### order_items
Contains line items for each order.

| Column | Type | Description |
|--------|------|-------------|
| item_id | INTEGER | Unique item identifier (Primary Key) |
| order_id | INTEGER | Reference to order (Foreign Key) |
| product_id | INTEGER | Reference to product (Foreign Key) |
| quantity | INTEGER | Number of units |
| price | DECIMAL(15,2) | Price per unit at time of order |

## Relationships
- customers (1) -> orders (N): One customer can have many orders
- orders (1) -> order_items (N): One order can have many items
- products (1) -> order_items (N): One product can be in many order items
