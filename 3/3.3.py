import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# Load dataset

adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame

# Features and target

X = df.drop("class", axis=1)
y = df["class"]

# 60 / 20 / 20 Stratified Split

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

# Feature Groups

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

numerical_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

# Preprocessing Pipeline

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numerical_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])

# Fit preprocessing ONLY on training data

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Subsample training set

rng = np.random.RandomState(42)

indices = rng.choice(
    X_train_processed.shape[0],
    size=3000,
    replace=False
)

X_sample = X_train_processed[indices]
y_sample = y_train.iloc[indices]

results = []

C_values = [0.001, 0.01, 0.1, 1, 10, 100]

for C in C_values:

    svm = SVC(
        kernel="linear",
        C=C,
        random_state=42
    )

    svm.fit(
        X_sample,
        y_sample
    )

    y_pred = svm.predict(
        X_val_processed
    )

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    f1 = f1_score(
        y_val,
        y_pred,
        pos_label=">50K"
    )

    n_support = svm.n_support_.sum()

    results.append({
        "C": C,
        "Accuracy": accuracy,
        "F1": f1,
        "Support Vectors": n_support
    })

results_df = pd.DataFrame(results)

print("\n=== Linear SVM: C Comparison ===")
print(results_df.round(4).to_string(index=False))

# Plot

plt.figure(figsize=(8, 6))

plt.plot(
    results_df["C"],
    results_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["C"],
    results_df["F1"],
    marker="s",
    label="F1 Score"
)

plt.xscale("log")

plt.xlabel("C")
plt.ylabel("Score")
plt.title("Linear SVM Performance vs C")
plt.legend()

plt.show()