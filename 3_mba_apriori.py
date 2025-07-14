import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("D:/Dataset/fyp2/merged_instacart_data.csv")
basket = df.groupby('order_id')['product_name'].apply(list).sample(n=10000, random_state=42)
te = TransactionEncoder()
df_encoded = pd.DataFrame(te.fit_transform(basket), columns=te.columns_)

# Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Output
rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]\
    .sort_values(by='lift', ascending=False)\
    .to_csv("apriori_product_rules.csv", index=False)

print("✅ Apriori product association rules saved to apriori_product_rules.csv")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=rules,
    x='support',
    y='confidence',
    size='lift',
    hue='lift',
    palette='coolwarm',
    sizes=(60, 300),
    legend=True
)
plt.title('Support vs Confidence (Lift as size & color)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.grid(True)
plt.show()

