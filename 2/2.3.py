import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

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

# Depth Experiment

depths = [2, 4, 6, 8, 10, 12, 15]

results = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    f1_scorer = make_scorer(
        f1_score,
        pos_label=">50K"
    )

    cv_results = cross_validate(
        model,
        X_train_processed,
        y_train,
        cv=5,
        scoring={
            "accuracy": "accuracy",
            "f1": f1_scorer
        }
    )

    model.fit(
        X_train_processed,
        y_train
    )

    y_train_pred = model.predict(
        X_train_processed
    )

    y_val_pred = model.predict(
        X_val_processed
    )

    results.append({
        "Depth": depth,
        "Train Accuracy": accuracy_score(
            y_train,
            y_train_pred
        ),
        "Validation Accuracy": accuracy_score(
            y_val,
            y_val_pred
        ),
        "Validation F1": f1_score(
            y_val,
            y_val_pred,
            pos_label=">50K"
        ),
        "CV Accuracy Mean": cv_results[
            "test_accuracy"
        ].mean(),
        "CV Accuracy Std": cv_results[
            "test_accuracy"
        ].std(),
        "CV F1 Mean": cv_results[
            "test_f1"
        ].mean(),
        "CV F1 Std": cv_results[
            "test_f1"
        ].std()
    })

# Results

results_df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\n=== Decision Tree Depth Comparison ===")
print(results_df.round(4))

# Best Depth by Validation F1

best_row = results_df.loc[
    results_df["Validation F1"].idxmax()
]

print("\n=== Best Depth ===")
print(best_row)