import pandas as pd
import time
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from itertools import combinations

# Load dataset
df = pd.read_csv("D:/Dataset/fyp2/merged_instacart_data.csv")

# Step 1: Prepare basket data at order level (by department)
basket_data = df.groupby('order_id')['department'].apply(list).sample(n=100000, random_state=42)
te = TransactionEncoder()
df_encoded = pd.DataFrame(te.fit_transform(basket_data), columns=te.columns_)

# Step 2: General evaluation function (Apriori, FP-Growth)
def evaluate_algorithm(name, method_func, **kwargs):
    start = time.time()
    freq_itemsets = method_func(df_encoded, **kwargs)
    duration = time.time() - start

    if freq_itemsets.empty:
        return {
            'Algorithm': name,
            'Num Rules': 0,
            'Avg Support': 0,
            'Avg Confidence': 0,
            'Avg Lift': 0,
            'Time (s)': round(duration, 4)
        }

    rules = association_rules(freq_itemsets, metric="lift", min_threshold=1.0)
    return {
        'Algorithm': name,
        'Num Rules': len(rules),
        'Avg Support': round(rules['support'].mean(), 6),
        'Avg Confidence': round(rules['confidence'].mean(), 6),
        'Avg Lift': round(rules['lift'].mean(), 6),
        'Time (s)': round(duration, 4)
    }

# Step 3: Enhanced ECLAT Evaluation (2-item rules)
def evaluate_eclat(df_encoded, min_support=0.05):
    start_time = time.time()
    num_transactions = df_encoded.shape[0]

    item_support = df_encoded.mean()
    frequent_items = item_support[item_support >= min_support]
    pairs = list(combinations(frequent_items.index, 2))

    results = []

    for item_a, item_b in pairs:
        support_a = item_support[item_a]
        support_b = item_support[item_b]
        support_ab = (df_encoded[item_a] & df_encoded[item_b]).sum() / num_transactions

        if support_ab >= min_support:
            confidence_ab = support_ab / support_a
            confidence_ba = support_ab / support_b
            lift = support_ab / (support_a * support_b)

            results.append({
                'Antecedent': item_a,
                'Consequent': item_b,
                'Support': round(support_ab, 6),
                'Confidence': round(confidence_ab, 6),
                'Lift': round(lift, 6)
            })
            results.append({
                'Antecedent': item_b,
                'Consequent': item_a,
                'Support': round(support_ab, 6),
                'Confidence': round(confidence_ba, 6),
                'Lift': round(lift, 6)
            })

    elapsed = time.time() - start_time
    rules_df = pd.DataFrame(results)

    return {
        'Algorithm': 'ECLAT',
        'Num Rules': len(rules_df),
        'Avg Support': round(rules_df['Support'].mean(), 6) if not rules_df.empty else 0,
        'Avg Confidence': round(rules_df['Confidence'].mean(), 6) if not rules_df.empty else 0,
        'Avg Lift': round(rules_df['Lift'].mean(), 6) if not rules_df.empty else 0,
        'Time (s)': round(elapsed, 4)
    }

# Step 4: Run all algorithms
result_apriori = evaluate_algorithm("Apriori", apriori, min_support=0.01, use_colnames=True)
result_fpgrowth = evaluate_algorithm("FP-Growth", fpgrowth, min_support=0.01, use_colnames=True)
result_eclat = evaluate_eclat(df_encoded, min_support=0.05)

# Step 5: Combine & save result
results_df = pd.DataFrame([result_apriori, result_fpgrowth, result_eclat])
results_df.to_csv("mba_model_evaluation.csv", index=False)
print("MBA evaluation results saved to mba_model_evaluation.csv")
print(results_df)

