from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml

# Load dataset
adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame

# Missing values
for col in ["workclass", "occupation", "native-country"]:
    df[col] = df[col].astype("object")
    df[col] = df[col].fillna("Unknown")

# Group rare countries
threshold = len(df) * 0.005

country_counts = df["native-country"].value_counts()

rare_countries = country_counts[
    country_counts < threshold
].index

df["native-country"] = df["native-country"].replace(
    rare_countries,
    "Other"
)

# One-Hot Encoding
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

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded = encoder.fit_transform(df[categorical_features])

print(f"Original categorical features: {len(categorical_features)}")
print(f"Encoded features: {encoded.shape[1]}")