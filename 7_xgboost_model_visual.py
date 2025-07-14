import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve

# Load Excel data
df = pd.read_excel("machine learning data.xlsx")

# 1️ Top 15 Predicted Products
recommended = df[df['score'] >= 0.5].copy()
top_products = recommended['product_name'].value_counts().head(15)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')

plt.title("Top 15 Predicted Products (XGBoost)", fontsize=18, color='black')
plt.xlabel("Predicted Count", fontsize=16, color='black')
plt.ylabel("Product Name", fontsize=16, color='black')
plt.xticks(fontsize=16, color='black')
plt.yticks(fontsize=16, color='black')

# Add value labels
for i, v in enumerate(top_products.values):
    plt.text(v + 1, i, str(v), color='black', va='center', fontsize=12)

plt.tight_layout()
plt.show()

# 2️ Top Recommended Departments
top_departments = recommended['department'].value_counts().head(10)

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=top_departments.values, y=top_departments.index, palette='coolwarm')

plt.title("Top Departments by Predicted Reorders", fontsize=18, color='black')
plt.xlabel("Predicted Count", fontsize=16, color='black')
plt.ylabel("Department", fontsize=16, color='black')
plt.xticks(fontsize=16, color='black')
plt.yticks(fontsize=16, color='black')

# Add value labels
for i, v in enumerate(top_departments.values):
    plt.text(v + 1, i, str(v), color='black', va='center', fontsize=12)

plt.tight_layout()
plt.show()

# 3 Score Distribution by Department
filtered_df = df[(df['score'] >= 0.5) & (df['department'] != 'missing')]

plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df, x='department', y='score')

plt.title("Score Distribution of Recommended Products by Department", fontsize=18, color='black')
plt.xlabel("Department", fontsize=14, color='black')
plt.ylabel("Score", fontsize=14, color='black')
plt.xticks(rotation=45, fontsize=14, color='black')
plt.yticks(fontsize=14, color='black')

plt.tight_layout()
plt.show()


# 4 Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, df['score'])

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='darkorange')
plt.title("Precision-Recall Curve", fontsize=18, color='black')
plt.xlabel("Recall", fontsize=14, color='black')
plt.ylabel("Precision", fontsize=14, color='black')
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.grid()
plt.tight_layout()
plt.show()
