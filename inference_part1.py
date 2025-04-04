import time

import xgboost
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading data
# ------------
print("Loading MNIST data...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(int)

# We will use just a subset to make training faster
X = X[:10000]
y = y[:10000]

# Let's train/split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler for LR
# ------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression
# ------------------
print("Training LR...")
log_clf = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)
log_clf.fit(X_train_scaled, y_train)

start = time.time()
y_pred_log = log_clf.predict(X_test_scaled)
log_time = time.time() - start
log_acc = accuracy_score(y_test, y_pred_log)

# Random Forest
# -------------
print("Training RF...")
rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf_clf.fit(X_train, y_train)

start = time.time()
y_pred_rf = rf_clf.predict(X_test)
rf_time = time.time() - start
rf_acc = accuracy_score(y_test, y_pred_rf)

# XGBoost with GPU
# ----------------
print("Training XGBoost...")
xgb_clf = xgboost.XGBClassifier(
    tree_method='gpu_hist',
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=0
)

try:
    xgb_clf.fit(X_train, y_train)
except xgboost.core.XGBoostError as e:
    print("XGBoost GPU failed, falling back to CPU !")
    xgb_clf = xgboost.XGBClassifier(
        tree_method='hist',
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0
    )
    xgb_clf.fit(X_train, y_train)

start = time.time()
y_pred_xgb = xgb_clf.predict(X_test)
xgb_time = time.time() - start
xgb_acc = accuracy_score(y_test, y_pred_xgb)

# Results
# ------
print("\n--- Inference Time Comparison (on test set) ---")
print(f"Logistic Regression: {log_time:.4f} sec | Accuracy: {log_acc:.4f}")
print(f"Random Forest      : {rf_time:.4f} sec | Accuracy: {rf_acc:.4f}")
print(f"XGBoost            : {xgb_time:.4f} sec | Accuracy: {xgb_acc:.4f}")
