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
    accuracy_score,
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

# min_samples_leaf
results = []

for leaf in [1, 2, 5, 10, 20, 50]:

    model = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=leaf,
        random_state=42
    )

    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_val_processed)

    results.append({
        "min_samples_leaf": leaf,
        "Validation Accuracy":
            accuracy_score(y_val, y_pred),
        "Validation F1":
            f1_score(
                y_val,
                y_pred,
                pos_label=">50K"
            )
    })

leaf_results = pd.DataFrame(results)

print("\n=== Min Samples Leaf ===")
print(leaf_results.round(4).to_string(index=False))

print("\n=== Best Min Samples Leaf ===")
print(
    leaf_results.loc[
        leaf_results["Validation F1"].idxmax()
    ]
)

# ccp_alpha
results = []

for alpha in [0.0, 0.0001, 0.0005,
              0.001, 0.005, 0.01]:

    model = DecisionTreeClassifier(
        max_depth=12,
        ccp_alpha=alpha,
        random_state=42
    )

    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_val_processed)

    results.append({
        "ccp_alpha": alpha,
        "Validation Accuracy":
            accuracy_score(y_val, y_pred),
        "Validation F1":
            f1_score(
                y_val,
                y_pred,
                pos_label=">50K"
            )
    })

alpha_results = pd.DataFrame(results)

print("\n=== CCP Alpha ===")
print(alpha_results.round(4).to_string(index=False))