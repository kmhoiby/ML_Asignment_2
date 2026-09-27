from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd

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

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    f1_score
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


# Best model from Q2.4
unweighted_tree = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=5,
    ccp_alpha=0.0001,
    random_state=42
)

unweighted_tree.fit(X_train_processed, y_train)

y_pred_unweighted = unweighted_tree.predict(
    X_val_processed
)

# Balanced model
balanced_tree = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=5,
    ccp_alpha=0.0001,
    class_weight="balanced",
    random_state=42
)

balanced_tree.fit(X_train_processed, y_train)

y_pred_balanced = balanced_tree.predict(
    X_val_processed
)

print("\n=== Unweighted Tree ===")
print(
    "Precision:",
    round(
        precision_score(
            y_val,
            y_pred_unweighted,
            pos_label=">50K"
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_val,
            y_pred_unweighted,
            pos_label=">50K"
        ),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_val,
            y_pred_unweighted,
            pos_label=">50K"
        ),
        4
    )
)

print("\n=== Balanced Tree ===")
print(
    "Precision:",
    round(
        precision_score(
            y_val,
            y_pred_balanced,
            pos_label=">50K"
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_val,
            y_pred_balanced,
            pos_label=">50K"
        ),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_val,
            y_pred_balanced,
            pos_label=">50K"
        ),
        4
    )
)