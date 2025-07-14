import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("D:/Dataset/fyp2/merged_instacart_data.csv")
basket = df.groupby('order_id')['department'].apply(lambda x: list(set(x))).sample(n=100000, random_state=42)
te = TransactionEncoder()
df_encoded = pd.DataFrame(te.fit_transform(basket), columns=te.columns_)

# Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]\
    .sort_values(by='lift', ascending=False)\
    .to_csv("apriori_department_rules.csv", index=False)

print("Apriori department association rules saved to apriori_department_rules.csv")

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
    legend=True,
    edgecolor= 'none'
)

# Set title and labels with larger font size and black color
plt.title('Support vs Confidence', fontsize=18, color='black')
plt.xlabel('Support', fontsize=14, color='black')
plt.ylabel('Confidence', fontsize=14, color='black')

# Customize tick label font size and color
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')

# Grid for better visibility
plt.grid(True)

# Customize legend font and title color
legend = plt.legend()
for text in legend.get_texts():
    text.set_fontsize(12)
    text.set_color('black')
legend.get_title().set_fontsize(14)
legend.get_title().set_color('black')

plt.show()

