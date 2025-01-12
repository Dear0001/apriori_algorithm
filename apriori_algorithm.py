# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Step 1: Preprocess the Data
data = {
    'TransactionID': [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5],
    'Product': ['Apple', 'Banana', 'Apple', 'Citrus', 'Banana', 'Apple', 'Citrus', 'Banana', 'Citrus', 'Apple', 'Apple', 'Banana']
}

# Convert the dataset into a DataFrame
df = pd.DataFrame(data)

# Transform the data into a one-hot encoded format
basket = df.groupby(['TransactionID', 'Product'])['Product'].count().unstack().fillna(0)

# Convert counts to binary values (1 for presence, 0 for absence)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Step 2: Find Frequent Itemsets
# Use the apriori() function with a minimum support threshold
min_support = 0.4  # Adjust this threshold as needed
frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

# Display frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Step 3: Generate Association Rules
# Use association_rules() to generate rules with a minimum confidence threshold
min_confidence = 0.7  # Adjust this threshold as needed
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=10)

# Display association rules
print("\nAssociation Rules:")
print(rules)

# Step 4: Filter and Interpret Rules
# Filter rules for specific products (e.g., Apple or Citrus)
filtered_rules = rules[
    (rules['antecedents'].apply(lambda x: 'Apple' in x)) |
    (rules['consequents'].apply(lambda x: 'Apple' in x)) |
    (rules['antecedents'].apply(lambda x: 'Citrus' in x)) |
    (rules['consequents'].apply(lambda x: 'Citrus' in x))
]

# Display filtered rules
print("\nFiltered Rules for Apple and Citrus:")
print(filtered_rules)

# Interpret and use these rules for product recommendations
# Example: Recommend products based on the rules
if not filtered_rules.empty:
    print("\nProduct Recommendations:")
    for index, rule in filtered_rules.iterrows():
        print(f"If a customer buys {rule['antecedents']}, they are likely to buy {rule['consequents']} (Confidence: {rule['confidence']:.2f})")
else:
    print("\nNo specific recommendations found for Apple or Citrus.")