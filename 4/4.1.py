import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Linear SVM",
        "RBF SVM"
    ],
    "Validation Accuracy": [
        0.8623,
        0.8531,
        0.8491
    ],
    "Validation F1": [
        0.6941,
        0.6535,
        0.6344
    ],
    "ROC-AUC": [
        0.8980,  # beste ROC-AUC du fant
        0.8980,
        0.8924
    ]
})

# Figure with all three metrics
ax = results.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.ylabel("Score")
plt.title("Validation Performance Comparison")
plt.ylim(0, 1)
plt.legend(loc="best")

plt.tight_layout()
plt.show()

