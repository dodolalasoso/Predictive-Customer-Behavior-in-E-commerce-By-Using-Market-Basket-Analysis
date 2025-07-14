import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Step 1: Load data
df = pd.read_csv("merged_instacart_data.csv")

# Step 2: Identify last order per user
last_orders = df.groupby('user_id')['order_number'].max().reset_index()
last_orders.columns = ['user_id', 'last_order_number']
df = df.merge(last_orders, on='user_id', how='left')
df['is_last_order'] = df['order_number'] == df['last_order_number']

# Step 3: Create labels
labels = df[df['is_last_order']][['user_id', 'product_id']].copy()
labels['label'] = 1 

# Step 4: Prepare historical training data (excluding last orders)
train_df = df[~df['is_last_order']].copy()

# Step 5: Feature Engineering

# User features
user_features = train_df.groupby('user_id').agg({
    'order_number': 'max',
    'days_since_prior_order': 'mean',
    'product_id': 'count'
}).reset_index()
user_features.columns = ['user_id', 'u_total_orders', 'u_avg_days_between_orders', 'u_total_products']

# Product features
product_features = train_df.groupby('product_id').agg({
    'reordered': ['sum', 'count']
})
product_features.columns = ['p_total_reorders', 'p_total_orders']
product_features['p_reorder_rate'] = product_features['p_total_reorders'] / product_features['p_total_orders']
product_features.reset_index(inplace=True)

# User-product interaction features
user_product = train_df.groupby(['user_id', 'product_id']).agg({
    'order_number': ['count', 'min', 'max'],
    'reordered': 'sum'
})
user_product.columns = ['up_order_count', 'up_first_order', 'up_last_order', 'up_reorders']
user_product['up_reorder_rate'] = user_product['up_reorders'] / user_product['up_order_count']
user_product.reset_index(inplace=True)

# Step 6: Combine all features
features = user_product.merge(user_features, on='user_id', how='left')
features = features.merge(product_features, on='product_id', how='left')
features = features.merge(labels, on=['user_id', 'product_id'], how='left')
features['label'] = features['label'].fillna(0).astype(int)

# Step 7: Train-test split
X = features.drop(columns=['user_id', 'product_id', 'label'])
y = features['label']
X = X.fillna(0)  # handle any missing values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Step 9: Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("AUC Score:", round(roc_auc_score(y_test, y_proba), 4))

# Step 10: Predict top items per customer (with product name)
features_test = features.iloc[X_test.index].copy()
features_test['score'] = y_proba

# Merge with product name
products_df = pd.read_csv("merged_instacart_data.csv") # ensure the file path is correct
top_predictions = features_test.merge(products_df[['product_id', 'product_name']], on='product_id', how='left')

# Filter predictions with score ≥ 0.5 and sort
top_predictions = top_predictions[top_predictions['score'] >= 0.5]
top_predictions = top_predictions.sort_values(by=['user_id', 'score'], ascending=[True, False])

# Save to file
top_predictions.to_csv("xgboost_predicted_products_with_name.csv", index=False)
print("Predicted product list with product_name saved to xgboost_predicted_products_with_name.csv")
