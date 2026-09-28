import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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

print(
    "\nExplained variance PC1:",
    round(explained_variance[0], 4)
)

print(
    "Explained variance PC2:",
    round(explained_variance[1], 4)
)

print(
    "Total explained variance:",
    round(explained_variance.sum(), 4)
)

# Encode target

y_train_num = (
    y_train == ">50K"
).astype(int)

y_val_num = (
    y_val == ">50K"
).astype(int)

# Best Models

decision_tree = DecisionTreeClassifier(
    max_depth=15,
    min_samples_leaf=1,
    ccp_alpha=0.0001,
    random_state=42
)

linear_svm = SVC(
    kernel="linear",
    C=1,
    random_state=42
)

rbf_svm = SVC(
    kernel="rbf",
    C=10,
    gamma=0.01,
    random_state=42
)

models = {
    "Decision Tree": decision_tree,
    "Linear SVM": linear_svm,
    "RBF SVM": rbf_svm
}

# Decision Boundary Function

def plot_boundary(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    ax,
    title
):

    model.fit(
        X_train,
        y_train
    )

    x_min, x_max = (
        X_val[:, 0].min() - 1,
        X_val[:, 0].max() + 1
    )

    y_min, y_max = (
        X_val[:, 1].min() - 1,
        X_val[:, 1].max() + 1
    )

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
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

    scatter = ax.scatter(
        X_val[:, 0],
        X_val[:, 1],
        c=y_val,
        cmap="coolwarm",
        edgecolor="k",
        s=20
    )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

# Plot

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

for ax, (name, model) in zip(
    axes,
    models.items()
):

    plot_boundary(
        model,
        X_train_pca,
        y_train_num,
        X_val_pca,
        y_val_num,
        ax,
        name
    )

plt.tight_layout()
plt.show()