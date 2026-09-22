import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame

# Replace "?" with NaN in the dataset.
#df.replace("?", np.nan, inplace=True)

# Cound missing values
missing_values = df.isnull().sum()

# Percentage missing
missing_percent = (df.isnull().sum() / len(df)) * 100

# Creating table overview
missing_table = pd.DataFrame({
    "Missing Count": missing_values,
    "Missing (%)": missing_percent
})

# Show only columns with missing values
print(missing_table.loc[missing_table["Missing Count"] > 0])

# Comparing occupation with workclass
occupation_table = pd.crosstab(
df["occupation"].isnull(),
        df["workclass"].isnull(),
        margins=True
)

occupation_table.index = [
    "Occupation Present",
    "Occupation Missing",
    "Total"
]

occupation_table.columns = [
    "Workclass Present",
    "Workclass Missing",
    "Total"
]
print("\nMissing: occupation vs workclass")
print(occupation_table)

# Compare native-country with workclass
native_country_table = pd.crosstab(
df["native-country"].isnull(),
        df["workclass"].isnull(),
        margins=True
)

native_country_table.index = [
    "Native-country Present",
    "Native-country Missing",
    "Total"
]

native_country_table.columns = [
    "Workclass Present",
    "Workclass Missing",
    "Total"
]
print("\nMissing: native-country vs workclass")
print(native_country_table)

plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values")
plt.show()