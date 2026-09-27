import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc
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
    size=10000,
    replace=False
)

X_sample = X_train_processed[indices]
y_sample = y_train.iloc[indices]

# Linear SVM

svm = SVC(
    kernel="linear",
    probability=True,
    random_state=42
)

# 5-fold CV

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracies = []

plt.figure(figsize=(8, 6))

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_sample, y_sample),
    start=1
):

    X_fold_train = X_sample[train_idx]
    X_fold_test = X_sample[test_idx]

    y_fold_train = y_sample.iloc[train_idx]
    y_fold_test = y_sample.iloc[test_idx]

    svm.fit(
        X_fold_train,
        y_fold_train
    )

    y_pred = svm.predict(
        X_fold_test
    )

    acc = accuracy_score(
        y_fold_test,
        y_pred
    )

    accuracies.append(acc)

    y_prob = svm.predict_proba(
        X_fold_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_fold_test,
        y_prob,
        pos_label=">50K"
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"Fold {fold} (AUC={roc_auc:.3f})"
    )

# ROC plot

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Linear SVM ROC Curves")
plt.legend()
plt.show()

# Mean/std accuracy

print(
    f"Accuracy Mean: {np.mean(accuracies):.4f}"
)

print(
    f"Accuracy Std: {np.std(accuracies):.4f}"
)