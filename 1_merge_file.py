import pandas as pd

# read csv file
orders = pd.read_csv('orders.csv')
order_products = pd.read_csv('order_products__prior.csv')
products = pd.read_csv('products.csv')
departments = pd.read_csv('departments.csv')

products_with_departments = pd.merge(products, departments, on="department_id", how="left")
order_details = pd.merge(order_products, orders, on="order_id", how="left")
final_data = pd.merge(order_details, products_with_departments, on="product_id", how="left")
final_data = final_data.dropna()

# save merged file
final_data.to_csv("merged_instacart_data.csv", index=False)
print(final_data.head())
