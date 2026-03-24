import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


# 1. Load data
# Replace this file name with your Kaggle CSV file name
df = pd.read_csv("Credit_Card.csv")


# 2. Basic cleanup

# 1 = default, 0 = no default
# Change this line if your dataset uses another target column name
target_col = "default.payment.next.month"

# Keep only rows where target exists
df = df.dropna(subset=[target_col])

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]


# 3. Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


# 4. Preprocessing
# Numeric: fill missing values + scale
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical: fill missing values + one-hot encode
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 6. Logistic regression model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
print("fitting model")
model.fit(X_train, y_train)
print("done")

# 7. Predictions and performance
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]   # predicted default probability (PD)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Model Performance")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Add PD to test set
results = X_test.copy()
results["Actual Default"] = y_test.values
results["Predicted Default"] = y_pred
results["PD"] = y_prob


# 9. Create LGD and EAD assumptions

# LGD = Loss Given Default
# EAD = Exposure at Default
#
# If your dataset already has loan amount, use it.
# If not, replace with a constant or another relevant column.

if "loan_amnt" in results.columns:
    results["EAD"] = results["loan_amnt"]
elif "loan_amount" in results.columns:
    results["EAD"] = results["loan_amount"]
else:
    results["EAD"] = 10000  # fallback assumption

# 60% loss if borrower defaults
results["LGD"] = 0.60


# 10. Expected Loss calculation
# Expected Loss = PD × LGD × EAD
results["Expected Loss"] = results["PD"] * results["LGD"] * results["EAD"]

print("\nSample Credit Risk Results:")
print(results[["PD", "LGD", "EAD", "Expected Loss"]].head().round(2))


# 11. Portfolio summary
total_expected_loss = results["Expected Loss"].sum()
avg_pd = results["PD"].mean()
avg_expected_loss = results["Expected Loss"].mean()

print("\nPortfolio Summary")
print(f"Number of loans scored: {len(results)}")
print(f"Average PD: {avg_pd:.4f}")
print(f"Average Expected Loss: ${avg_expected_loss:,.2f}")
print(f"Total Expected Loss: ${total_expected_loss:,.2f}")


# 12. Risk banding / scorecard style buckets
results["Risk Band"] = pd.cut(
    results["PD"],
    bins=[0, 0.05, 0.10, 0.20, 1.00],
    labels=["Low", "Moderate", "High", "Very High"],
    include_lowest=True
)

band_summary = results.groupby("Risk Band", observed=False).agg(
    Count=("PD", "count"),
    Avg_PD=("PD", "mean"),
    Avg_EL=("Expected Loss", "mean"),
    Total_EL=("Expected Loss", "sum")
).reset_index()

print("\nRisk Band Summary")
print(band_summary.round(2))


# 13. Graph 1: Histogram of PD
plt.figure(figsize=(8, 5))
plt.hist(results["PD"], bins=25)
plt.xlabel("Predicted Default Probability (PD)")
plt.ylabel("Number of Loans")
plt.title("Distribution of Predicted Default Probabilities")
plt.tight_layout()
plt.show()


# 14. Graph 2: Expected Loss by Risk Band
plt.figure(figsize=(8, 5))
plt.bar(band_summary["Risk Band"].astype(str), band_summary["Total_EL"])
plt.xlabel("Risk Band")
plt.ylabel("Total Expected Loss ($)")
plt.title("Total Expected Loss by Risk Band")
plt.tight_layout()
plt.show()


# Export results
results.to_csv("credit_risk_results.csv", index=False)
band_summary.to_csv("credit_risk_band_summary.csv", index=False)