import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OrdinalEncoder

# ANOVA F-test

# Load dataset
adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame

# Target
y = df["class"]

# Numerical features
numerical_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

X = df[numerical_features]

# ANOVA F-test
f_scores, p_values = f_classif(X, y)

results = pd.DataFrame({
    "Feature": numerical_features,
    "F-Score": f_scores,
    "P-Value": p_values
})

results = results.sort_values(
    "F-Score",
    ascending=False
)

print("\n=== Numerical Feature Significance ===")
print(results)

# Chi-Square test

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

# Handle missing values
for col in ["workclass", "occupation", "native-country"]:
    df[col] = df[col].astype("object")
    df[col] = df[col].fillna("Unknown")

encoder = OrdinalEncoder()

X_cat = encoder.fit_transform(
    df[categorical_features]
)

chi_scores, p_values = chi2(
    X_cat,
    y
)

cat_results = pd.DataFrame({
    "Feature": categorical_features,
    "Chi2 Score": chi_scores,
    "P-Value": p_values
})

print("\n=== Categorical Feature Significance ===")
print(
    cat_results.sort_values(
        "Chi2 Score",
        ascending=False
    )
)

# Combine og find topp 8

all_results = pd.concat([
    pd.DataFrame({
        "Feature": numerical_features,
        "Score": f_scores
    }),
    pd.DataFrame({
        "Feature": categorical_features,
        "Score": chi_scores
    })
])

top8 = all_results.sort_values(
    "Score",
    ascending=False
).head(8)

print("\n=== Top 8 Features ===")
print(top8)