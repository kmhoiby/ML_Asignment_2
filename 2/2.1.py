from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    cross_validate
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
    balanced_accuracy_score
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

# Fit preprocessing only on training data

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

# Default Decision Tree

dt = DecisionTreeClassifier(
    random_state=42
)

# 5-Fold Cross Validation

cv_results = cross_validate(
    dt,
    X_train_processed,
    y_train,
    cv=5,
    scoring=[
        "accuracy",
        "balanced_accuracy"
    ]
)

print("\n=== 5-Fold Cross Validation ===")

print(
    f"Accuracy: "
    f"{cv_results['test_accuracy'].mean():.4f} "
    f"+/- "
    f"{cv_results['test_accuracy'].std():.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{cv_results['test_balanced_accuracy'].mean():.4f} "
    f"+/- "
    f"{cv_results['test_balanced_accuracy'].std():.4f}"
)

# Train on Entire Training Set

dt.fit(
    X_train_processed,
    y_train
)

# Validation Evaluation

y_pred = dt.predict(
    X_val_processed
)

validation_accuracy = accuracy_score(
    y_val,
    y_pred
)

validation_balanced_accuracy = balanced_accuracy_score(
    y_val,
    y_pred
)

print("\n=== Validation Set Results ===")

print(
    f"Validation Accuracy: "
    f"{validation_accuracy:.4f}"
)

print(
    f"Validation Balanced Accuracy: "
    f"{validation_balanced_accuracy:.4f}"
)