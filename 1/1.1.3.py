import pandas as pd
from sklearn.datasets import fetch_openml

# Load dataset
adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame

# Fill missing values from Q1.1
for col in ["workclass", "occupation", "native-country"]:
    df[col] = df[col].astype("object")
    df[col] = df[col].fillna("Unknown")

# Identify categorical features
categorical_features = df.select_dtypes(
    include=["category", "object"]
).columns

# Calculate cardinality
cardinality_table = pd.DataFrame({
    "Feature": categorical_features,
    "Cardinality": [
        df[col].nunique()
        for col in categorical_features
    ]
})

print("\n=== Cardinality of Categorical Features ===")
print(
    cardinality_table
    .sort_values("Cardinality", ascending=False)
)

# --------------------------------------------------
# High-cardinality feature: native-country
# Group categories representing less than 1%
# --------------------------------------------------

threshold = len(df) * 0.005

print(
    f"\nFrequency threshold: "
    f"{int(threshold)} observations (1%)"
)

# Original category count
original_count = df["native-country"].nunique()

# Count observations per country
country_counts = (
    df["native-country"]
    .value_counts()
)

# Find rare categories
rare_countries = country_counts[
    country_counts < threshold
].index

print(
    f"\nRare categories identified: "
    f"{len(rare_countries)}"
)

# Create grouped feature
df["native-country-grouped"] = (
    df["native-country"]
    .replace(rare_countries, "Other")
)

grouped_count = (
    df["native-country-grouped"]
    .nunique()
)

print("\n=== Category Count Comparison ===")
print(f"Original categories: {original_count}")
print(f"Grouped categories:  {grouped_count}")

print("\n=== Category Frequencies After Grouping ===")
print(
    df["native-country-grouped"]
    .value_counts()
)