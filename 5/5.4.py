import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.decomposition import PCA

# Load Dataset

adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame

X = df.drop("class", axis=1)
y = df["class"]

# Train / Validation / Test Split

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

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# PCA (2D)

pca = PCA(
    n_components=2,
    random_state=42
)

X_train_pca = pca.fit_transform(
    X_train_processed.toarray()
)

X_val_pca = pca.transform(
    X_val_processed.toarray()
)

explained_variance = (
    pca.explained_variance_ratio_
)

# Encode target

y_train_num = (
    y_train == ">50K"
).astype(int)

y_val_num = (
    y_val == ">50K"
).astype(int)

rng = np.random.RandomState(42)

indices = rng.choice(
X_train_pca.shape[0],
size=3000,
replace=False
)

X_train_pca_sample = X_train_pca[indices]
y_train_num_sample = y_train_num.iloc[indices]

print(
    "Total explained variance:",
    round(
        explained_variance.sum(),
        4
    )
)

# 5.4

depths = [2, 4, 8, 15]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes = axes.ravel()

for ax, depth in zip(axes, depths):

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(
        X_train_pca_sample,
        y_train_num_sample
    )

    x_min, x_max = (
        X_val_pca[:, 0].min() - 1,
        X_val_pca[:, 0].max() + 1
    )

    y_min, y_max = (
        X_val_pca[:, 1].min() - 1,
        X_val_pca[:, 1].max() + 1
    )

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 150),
        np.linspace(y_min, y_max, 150)
    )

    Z = model.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        cmap="coolwarm"
    )

    ax.scatter(
        X_val_pca[:, 0],
        X_val_pca[:, 1],
        c=y_val_num,
        cmap="coolwarm",
        edgecolor="k",
        s=15
    )

    ax.set_title(
        f"max_depth={depth}"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.tight_layout()
plt.show()