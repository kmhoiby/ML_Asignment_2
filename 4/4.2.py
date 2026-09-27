from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    fbeta_score
)

import numpy as np

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

# 60 / 20 / 20 split

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

# Feature groups

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

# Preprocessing pipeline

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

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_val_processed = preprocessor.transform(
    X_val
)

# Subsample for SVM

rng = np.random.RandomState(42)

indices = rng.choice(
    X_train_processed.shape[0],
    size=3000,
    replace=False
)

X_sample = X_train_processed[indices]
y_sample = y_train.iloc[indices]

# Best Decision Tree

decision_tree = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=5,
    ccp_alpha=0.0001,
    random_state=42
)

decision_tree.fit(
    X_train_processed,
    y_train
)

dt_pred = decision_tree.predict(
    X_val_processed
)

# Best Linear SVM

linear_svm = SVC(
    kernel="linear",
    C=1,
    random_state=42
)

linear_svm.fit(
    X_sample,
    y_sample
)

svm_pred = linear_svm.predict(
    X_val_processed
)

# F-Beta Scores

dt_f05 = fbeta_score(
    y_val,
    dt_pred,
    beta=0.5,
    pos_label=">50K"
)

dt_f2 = fbeta_score(
    y_val,
    dt_pred,
    beta=2,
    pos_label=">50K"
)

svm_f05 = fbeta_score(
    y_val,
    svm_pred,
    beta=0.5,
    pos_label=">50K"
)

svm_f2 = fbeta_score(
    y_val,
    svm_pred,
    beta=2,
    pos_label=">50K"
)

# Print results

print("\n=== F-Beta Comparison ===")

print(
    f"Decision Tree: "
    f"F0.5 = {dt_f05:.4f}, "
    f"F2 = {dt_f2:.4f}"
)

print(
    f"Linear SVM: "
    f"F0.5 = {svm_f05:.4f}, "
    f"F2 = {svm_f2:.4f}"
)