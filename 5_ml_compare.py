# compare_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Read csv file
df = pd.read_csv("merged_instacart_data.csv")

# Last order for every users
last_orders = df.groupby('user_id')['order_number'].max().reset_index()
last_orders.columns = ['user_id', 'last_order_number']
df = df.merge(last_orders, on='user_id', how='left')
df['is_last_order'] = df['order_number'] == df['last_order_number']


labels = df[df['is_last_order']][['user_id', 'product_id']].copy()
labels['label'] = 1
train_df = df[~df['is_last_order']].copy()

# user features
user_features = train_df.groupby('user_id').agg({
    'order_number': 'max',
    'days_since_prior_order': 'mean',
    'product_id': 'count'
}).reset_index()
user_features.columns = ['user_id', 'u_total_orders', 'u_avg_days_between_orders', 'u_total_products']

# product features
product_features = train_df.groupby('product_id').agg({
    'reordered': ['sum', 'count']
})
product_features.columns = ['p_total_reorders', 'p_total_orders']
product_features['p_reorder_rate'] = product_features['p_total_reorders'] / product_features['p_total_orders']
product_features.reset_index(inplace=True)

# user-product features
user_product = train_df.groupby(['user_id', 'product_id']).agg({
    'order_number': ['count', 'min', 'max'],
    'reordered': 'sum'
})
user_product.columns = ['up_order_count', 'up_first_order', 'up_last_order', 'up_reorders']
user_product['up_reorder_rate'] = user_product['up_reorders'] / user_product['up_order_count']
user_product.reset_index(inplace=True)

# merge features
features = user_product.merge(user_features, on='user_id', how='left')
features = features.merge(product_features, on='product_id', how='left')
features = features.merge(labels, on=['user_id', 'product_id'], how='left')
features['label'] = features['label'].fillna(0).astype(int)
features.to_csv("training_features.csv", index=False)

# split test and train
X = features.drop(columns=['user_id', 'product_id', 'label'])
y = features['label']
X = X.fillna(0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
models = {
    'Logistic Regression': LogisticRegression(max_iter=3000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'KNN': KNeighborsClassifier()
}


results = []

for name, model in models.items():
    print(f"\n🧠 Training model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    report = classification_report(y_val, y_pred, output_dict=True)
    auc = roc_auc_score(y_val, y_proba) if y_proba is not None else None
    acc = accuracy_score(y_val, y_pred)

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(report['1']['precision'], 4),
        'Recall': round(report['1']['recall'], 4),
        'F1-Score': round(report['1']['f1-score'], 4),
        'AUC': round(auc, 4) if auc else "N/A"

    })
# results
result_df = pd.DataFrame(results)
print("\nModel Comparison Results:")
print(result_df)
result_df.to_csv("model_results.csv", index=False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
df = pd.read_csv("model_results.csv")

# Melt DataFrame
melted = df.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1-Score', 'AUC'],
                 var_name='Metric', value_name='Score')

# Set font size and color globally
plt.rcParams.update({
    'font.size': 14,         # Increase font size
    'text.color': 'black',   # Font color for text
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.edgecolor': 'black'
})

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=melted, x='Model', y='Score', hue='Metric')

plt.title("Model Comparison by Evaluation Metrics", fontsize=18, color='black')
plt.ylim(0, 1)
plt.ylabel("Score", fontsize=16, color='black')
plt.xlabel("Model", fontsize=16, color='black')
plt.xticks(fontsize=16, color='black')
plt.yticks(fontsize=16, color='black')
plt.legend(title="Metric", title_fontsize=14, fontsize=12)

plt.grid(True, axis='y')
plt.tight_layout()
plt.show()