import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load dataset
adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame

# Numerical features
numerical_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

# Correlation matrix
corr_matrix = df[numerical_features].corr()

print("\n=== Correlation Matrix ===")
print(corr_matrix.round(3).to_string())

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()