from scipy.stats import zscore
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame

# Numerical features to analyze
numerical_features = [
    "age",
    "hours-per-week",
    "capital-gain",
    "capital-loss"
]

# Dictionaries for storing results
iqr_results = {}
zscore_results = {}

print("\n=== IQR Outlier Detection ===")

for col in numerical_features:

    # Calculate quartiles
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect outliers
    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    iqr_results[col] = len(outliers)

    print(f"{col}: {len(outliers)} outliers")


print("\n=== Z-Score Outlier Detection ===")

for col in numerical_features:

    # Calculate z-scores
    z_scores = np.abs(zscore(df[col]))

    # Detect outliers
    outliers = df[z_scores > 3]

    zscore_results[col] = len(outliers)

    print(f"{col}: {len(outliers)} outliers")


# Summary table
outlier_summary = pd.DataFrame({
    "Feature": numerical_features,
    "IQR Outliers": [
        iqr_results[col]
        for col in numerical_features
    ],
    "Z-Score Outliers": [
        zscore_results[col]
        for col in numerical_features
    ]
})

print("\n=== Outlier Summary ===")
print(outlier_summary)


# Boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, col in zip(axes.flatten(), numerical_features):
    sns.boxplot(
        x=df[col],
        ax=ax
    )
    ax.set_title(col)

plt.tight_layout()
plt.show()