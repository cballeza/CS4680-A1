import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Dataset
df = pd.read_csv("Multiclass Diabetes Dataset.csv")

print("Columns in dataset:", df.columns.tolist())

target_class = "Class" #classification target
target_reg = "BMI" #regression target

# Features
features = [col for col in df.columns if col not in [target_class, target_reg]]

X = df[features]

# Targets
y_class = df[target_class]
y_reg = df[target_reg]


# Model Training
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# -- Models --

# Classification Model
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)
clf.fit(X_train_c, y_train_c)
y_pred_class = clf.predict(X_test_c)

# Regression Model
reg = make_pipeline(
    StandardScaler(),
    LinearRegression()
)
reg.fit(X_train_r, y_train_r)
y_pred_reg = reg.predict(X_test_r)

# -- Evaluation --

# --- Classification ---
acc = accuracy_score(y_test_c, y_pred_class)
prec = precision_score(y_test_c, y_pred_class, average="weighted")
rec = recall_score(y_test_c, y_pred_class, average="weighted")
f1 = f1_score(y_test_c, y_pred_class, average="weighted")

print("\nClassification Results:")
print(f"Accuracy: {acc:.3f} - Precision: {prec:.3f} - Recall: {rec:.3f} - F1: {f1:.3f}")

print("\nPredictions (Class labels):")
print(f"{'Actual':>8} {'Predicted':>12}")
for actual, pred in list(zip(y_test_c[:10], y_pred_class[:10])):
    print(f"{actual:8} {pred:12}")

# --- Regression ---
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_reg))
mae = mean_absolute_error(y_test_r, y_pred_reg)
r2 = r2_score(y_test_r, y_pred_reg)

print("\nRegression Results:")
print(f"RMSE: {rmse:,.2f} - MAE: {mae:,.2f} - R^2: {r2:.3f}")

print("\nPredictions (BMI values):")
print(f"{'Actual':>10} {'Predicted':>12}")
for actual, pred in list(zip(y_test_r[:10], y_pred_reg[:10])):
    print(f"{actual:10.2f} {pred:12.2f}")
