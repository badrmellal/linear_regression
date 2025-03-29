import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, \
    roc_curve, roc_auc_score

# we'll use iris dataset for a binary classification task
# we'll classify whether a flower is Iris-Setosa (class 0) or not (class 1)
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # Petal length and width
y = (iris["target"] == 1).astype(int)  # 1 if Versicolor, 0 if not

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binary Classification using different models

# 1. Linear SVC model
svm_clf = Pipeline([
    ("scaler", StandardScaler()),  # Scale features for better convergence
    ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
])
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)

# 2. SGD Classifier with hinge loss
sgd_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDClassifier(loss="hinge", learning_rate="constant", eta0=0.001,
                          penalty="l2", alpha=0.0001, max_iter=1000, random_state=42)),
])
sgd_clf.fit(X_train, y_train)
sgd_pred = sgd_clf.predict(X_test)

# 3. Logistic Regression
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(C=10, random_state=42)),
])
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)

# 4. Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

# Performance Evaluation

# Function to evaluate metrics for each classifier
def evaluate_classifier(clf, name, X_test, y_test):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"--- {name} Performance ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print()

    return precision, recall, f1


# Evaluate each classifier
svm_metrics = evaluate_classifier(svm_clf, "Linear SVM", X_test, y_test)
sgd_metrics = evaluate_classifier(sgd_clf, "SGD Classifier", X_test, y_test)
log_metrics = evaluate_classifier(log_reg, "Logistic Regression", X_test, y_test)
rf_metrics = evaluate_classifier(rf_clf, "Random Forest", X_test, y_test)


# ======== Decision Boundaries Visualization =========

# Function to plot decision boundaries
def plot_decision_boundaries(clfs, names, X, y):
    # Create a mesh grid to visualize the decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    plt.figure(figsize=(15, 4))

    for i, (clf, name) in enumerate(zip(clfs, names)):
        plt.subplot(1, 3, i + 1)

        # Plot the decision boundary
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)

        # Plot the original data points
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

        plt.title(name)
        plt.xlabel("Petal length")
        plt.ylabel("Petal width")

    plt.tight_layout()
    plt.show()


# Plot decision boundaries for all classifiers
plot_decision_boundaries([svm_clf, sgd_clf, log_reg],
                         ["Linear SVM", "SGD Classifier", "Logistic Regression"],
                         X, y)


#  ROC Curve and Precision-Recall Curve

def plot_roc_precision_recall_curves(clf, X_test, y_test, name):
    plt.figure(figsize=(12, 5))

    # Get the decision function scores or probabilities
    if hasattr(clf, "decision_function"):
        y_scores = clf.decision_function(X_test)
    else:
        y_scores = clf.predict_proba(X_test)[:, 1]

    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    plt.plot(recalls, precisions, label=name)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Plot ROC and Precision-Recall curves for Logistic Regression
plot_roc_precision_recall_curves(log_reg, X_test, y_test, "Logistic Regression")


# Stochastic Gradient Descent Implementation

# A simple implementation of SGD for logistic regression to understand how it works
class MySGDClassifier:
    def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=32, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=(n_features,))
        self.bias = 0.0

        # Training with SGD
        for _ in range(self.max_iter):
            # Generate random mini-batch indices
            indices = rng.permutation(n_samples)
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:min(i + self.batch_size, n_samples)]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Forward pass: compute predictions
                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._sigmoid(z)

                # Compute gradients (logistic regression loss)
                dw = np.dot(X_batch.T, (y_pred - y_batch)) / len(y_batch)
                db = np.sum(y_pred - y_batch) / len(y_batch)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# Train and evaluate our custom SGD implementation
my_sgd_clf = MySGDClassifier(learning_rate=0.01, max_iter=100, random_state=42)
my_sgd_clf.fit(StandardScaler().fit_transform(X_train), y_train)
my_sgd_pred = my_sgd_clf.predict(StandardScaler().fit_transform(X_test))

print("--- Custom SGD Implementation Performance ---")
print(f"Accuracy: {np.mean(my_sgd_pred == y_test):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, my_sgd_pred)}")
print(f"Precision: {precision_score(y_test, my_sgd_pred):.4f}")
print(f"Recall: {recall_score(y_test, my_sgd_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, my_sgd_pred):.4f}")