import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    f1_score,
    roc_auc_score
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

# Linear SVM

linear_svm = SVC(
    kernel="linear",
    random_state=42
)

linear_svm.fit(
    X_sample,
    y_sample
)


linear_pred = linear_svm.predict(
    X_val_processed
)

linear_scores = linear_svm.decision_function(
    X_val_processed
)

linear_accuracy = accuracy_score(
    y_val,
    linear_pred
)

linear_f1 = f1_score(
    y_val,
    linear_pred,
    pos_label=">50K"
)

linear_auc = roc_auc_score(
    y_val,
    linear_scores
)

# RBF SVM

rbf_svm = SVC(
    kernel="rbf",
    random_state=42
)

rbf_svm.fit(
    X_sample,
    y_sample
)

rbf_pred = rbf_svm.predict(
    X_val_processed
)

rbf_scores = rbf_svm.decision_function(
    X_val_processed
)

rbf_accuracy = accuracy_score(
    y_val,
    rbf_pred
)

rbf_f1 = f1_score(
    y_val,
    rbf_pred,
    pos_label=">50K"
)

rbf_auc = roc_auc_score(
    y_val,
    rbf_scores
)

# Results

print("\n=== Linear SVM ===")
print(f"Accuracy: {linear_accuracy:.4f}")
print(f"F1: {linear_f1:.4f}")
print(f"ROC-AUC: {linear_auc:.4f}")

print("\n=== RBF SVM ===")
print(f"Accuracy: {rbf_accuracy:.4f}")
print(f"F1: {rbf_f1:.4f}")
print(f"ROC-AUC: {rbf_auc:.4f}")


# ROC Curves
fpr_linear, tpr_linear, _ = roc_curve(
    y_val,
    linear_scores,
    pos_label=">50K"
)

fpr_rbf, tpr_rbf, _ = roc_curve(
    y_val,
    rbf_scores,
    pos_label=">50K"
)

auc_linear = roc_auc_score(
    y_val,
    linear_scores
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_linear,
    tpr_linear,
    label=f"Linear SVM (AUC={auc_linear:.3f})"
)

plt.plot(
    fpr_rbf,
    tpr_rbf,
    label=f"RBF SVM (AUC={rbf_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Linear vs RBF SVM ROC Curve")
plt.legend()

plt.show()
