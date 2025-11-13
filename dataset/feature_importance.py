import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt


# 1) LOAD THE CLEANED DATASET

# Figure out project root:

cwd = os.getcwd()
if os.path.basename(cwd) == "dataset":
    PROJECT_ROOT = os.path.dirname(cwd)
else:
    PROJECT_ROOT = cwd

data_path = os.path.join(PROJECT_ROOT, "data", "clean_df.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"[Error] Could not find file: {data_path}")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

TARGET = "injury_severity"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in clean_df.csv")

print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")


# 2) SEPARATE FEATURES/TARGET

df = df.dropna(subset=[TARGET])
X = df.drop(columns=[TARGET])
y = df[TARGET]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print(f"Using {len(cat_cols)} categorical features and {len(num_cols)} numerical features.")


# 3) RANDOM FOREST IMPORTANCE (FAST VERSION)

# To keep this fast to run:
#  - we train on a SAMPLE of the data (e.g. 20,000 rows)
#  - we use a smaller random forest (fewer trees + limited depth)

# 3.1) Take a random sample for RF training
MAX_ROWS_RF = 20000

if len(X) > MAX_ROWS_RF:
    sample_idx = np.random.RandomState(42).choice(len(X), size=MAX_ROWS_RF, replace=False)
    X_rf = X.iloc[sample_idx].copy()
    y_rf = y.iloc[sample_idx].copy()
    print(f"RandomForest: using a sample of {len(X_rf):,} rows (out of {len(X):,})")
else:
    X_rf = X.copy()
    y_rf = y.copy()
    print(f"RandomForest: using all {len(X_rf):,} rows (dataset is small enough)")

# 3.2) Preprocessing: one-hot for categoricals, impute for numericals
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols)
    ]
)

# 3.3) Smaller / faster RandomForest
rf = RandomForestClassifier(
    n_estimators=80,        # fewer trees than 200
    max_depth=8,            # limit depth for speed & regularization
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipe = Pipeline([("prep", preprocess), ("rf", rf)])

print("Fitting RandomForest (this may take a short moment)...")
pipe.fit(X_rf, y_rf)

# 3.4) Extract feature importances
# Get expanded feature names from one-hot encoder
ohe = pipe.named_steps["prep"].named_transformers_["cat"]
cat_expanded = ohe.get_feature_names_out(cat_cols)
feature_names = list(cat_expanded) + num_cols

importances = pipe.named_steps["rf"].feature_importances_

rf_results = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
      .sort_values("importance", ascending=False)
)

print("\n=== TOP 20 FEATURES (RandomForest) ===")
print(rf_results.head(20))



# 4) MUTUAL INFORMATION

print("\nComputing Mutual Information scores")

X_mi = X.copy()

# Factorize categoricals (convert to integer codes)
for c in cat_cols:
    X_mi[c], _ = pd.factorize(X_mi[c])

mi_scores = mutual_info_classif(X_mi, y, random_state=42)

mi_results = (
    pd.DataFrame({"feature": X_mi.columns, "importance": mi_scores})
      .sort_values("importance", ascending=False)
)

print("\n=== TOP 20 FEATURES (Mutual Information) ===")
print(mi_results.head(20))


# 5) SAVE RESULTS

rf_results.to_csv("feature_importance_random_forest.csv", index=False)
mi_results.to_csv("feature_importance_mutual_info.csv", index=False)

print("\nSaved:")
print(" - feature_importance_random_forest.csv")
print(" - feature_importance_mutual_info.csv")


# 6) PLOT RESULTS

top = rf_results.head(15).iloc[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top["feature"], top["importance"])
plt.title("Top Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
