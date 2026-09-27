import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

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
    accuracy_score,
    f1_score,
    make_scorer
)

# Load Dataset

adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame

# Features and Target

X = df.drop("class", axis=1)
y = df["class"]

# 60 / 20 / 20 Split

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

# Preprocessing

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

# SVM Subsample

sample_size = 3000

rng = pd.Series(range(X_train_processed.shape[0])).sample(
    n=sample_size,
    random_state=42
)

X_train_svm = X_train_processed[rng]
y_train_svm = y_train.iloc[rng]

# Decision Tree GridSearch
f1_scorer = make_scorer(
    f1_score,
    pos_label=">50K"
)

dt_param_grid = {
    "max_depth": [8, 10, 12, 15],
    "min_samples_leaf": [1, 5, 10],
    "ccp_alpha": [0.0, 0.0001, 0.001]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
    scoring=f1_scorer,
    cv=5,
    n_jobs=-1
)

dt_grid.fit(
    X_train_processed,
    y_train
)

# SVM GridSearch

svm_param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "gamma": [0.001, 0.01, 0.1],
    "kernel": ["rbf"]
}

svm_grid = GridSearchCV(
    SVC(),
    param_grid=svm_param_grid,
    scoring=f1_scorer,
    cv=5,
    n_jobs=-1
)

svm_grid.fit(
    X_train_svm,
    y_train_svm
)

# Best Models

best_dt = dt_grid.best_estimator_
best_svm = svm_grid.best_estimator_

# Validation Performance

dt_pred = best_dt.predict(
    X_val_processed
)

svm_pred = best_svm.predict(
    X_val_processed
)

dt_accuracy = accuracy_score(
    y_val,
    dt_pred
)

dt_f1 = f1_score(
    y_val,
    dt_pred,
    pos_label=">50K"
)

svm_accuracy = accuracy_score(
    y_val,
    svm_pred
)

svm_f1 = f1_score(
    y_val,
    svm_pred,
    pos_label=">50K"
)

# Results

print("\n=== Best Decision Tree ===")
print(dt_grid.best_params_)
print(
    f"Best CV F1: "
    f"{dt_grid.best_score_:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{dt_accuracy:.4f}"
)

print(
    f"Validation F1: "
    f"{dt_f1:.4f}"
)

print("\n=== Best SVM ===")
print(svm_grid.best_params_)
print(
    f"Best CV F1: "
    f"{svm_grid.best_score_:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{svm_accuracy:.4f}"
)

print(
    f"Validation F1: "
    f"{svm_f1:.4f}"
)

# Performance Plot

results = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "SVM"
    ],
    "Validation Accuracy": [
        dt_accuracy,
        svm_accuracy
    ],
    "Validation F1": [
        dt_f1,
        svm_f1
    ]
})

results.set_index("Model").plot(
    kind="bar",
    figsize=(8, 5)
)

plt.ylabel("Score")
plt.title("GridSearch Best Models")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()