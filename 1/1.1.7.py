import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# Load dataset
adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame

numerical_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

# Standardization
scaler = StandardScaler()

df_scaled = pd.DataFrame(
    scaler.fit_transform(df[numerical_features]),
    columns=numerical_features
)

# Plot before and after
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

features = ["age", "education-num",
            "capital-gain", "hours-per-week"]

for ax, feature in zip(axes.flatten(), features):

    sns.boxplot(
        data=pd.DataFrame({
            "Original": df[feature],
            "Standardized": df_scaled[feature]
        }),
        ax=ax
    )

    ax.set_title(feature)

plt.tight_layout()
plt.show()

# Compare means and standard deviations

original_stats = pd.DataFrame({
    "Mean": df[numerical_features].mean(),
    "Std": df[numerical_features].std()
})

scaled_stats = pd.DataFrame({
    "Mean": df_scaled.mean(),
    "Std": df_scaled.std()
})

print("\n=== Original Statistics ===")
print(original_stats)

print("\n=== Standardized Statistics ===")
print(scaled_stats)