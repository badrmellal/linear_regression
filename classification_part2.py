import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import check_random_state
import time

# Load and Prepare MNIST Dataset
print("Loading MNIST dataset...")
# Fetch the MNIST dataset from OpenML
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist["data"], mnist["target"]

# Convert string labels to integers
y = y.astype(np.uint8)

# Split the dataset into training and test sets
# We'll use a smaller training set for faster computation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, train_size=30000, random_state=42
)

# Convert pandas DataFrame to numpy array because i had an error loading data
if hasattr(X_train, 'values'):  # Check if it's a pandas DataFrame
    X_train_array = X_train.values
    X_test_array = X_test.values
else:
    X_train_array = X_train
    X_test_array = X_test

# Visualize Some Examples
def plot_digit(image_data):
    """Plot a single digit from the MNIST dataset."""
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

# Plot a few examples
plt.figure(figsize=(15, 5))
for i in range(10):
    # Find indices where y_train equals i
    digit_indices = np.where(y_train == i)[0]
    if len(digit_indices) > 0:
        # Get the first image with this label
        first_image = X_train_array[digit_indices[0]]
        plt.subplot(2, 5, i + 1)
        plot_digit(first_image)
        plt.title(f"Digit: {i}")
plt.tight_layout()
plt.show()



# Multiclass Classification Models

# First we scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_array)
X_test_scaled = scaler.transform(X_test_array)

# Model 1: SGD Classifier
print("Training SGD Classifier...")
start_time = time.time()
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_scaled, y_train)
sgd_time = time.time() - start_time
print(f"SGD Classifier trained in {sgd_time:.2f} seconds")

# Model 2: SVM Classifier
print("Training SVM Classifier...")
start_time = time.time()
# Here we will use smaller subset for SVM due to computational constraints
X_train_small, _, y_train_small, _ = train_test_split(
    X_train_scaled, y_train, train_size=10000, random_state=42
)
svm_clf = SVC(gamma='scale', random_state=42)
svm_clf.fit(X_train_small, y_train_small)
svm_time = time.time() - start_time
print(f"SVM Classifier trained in {svm_time:.2f} seconds")

# Model 3: Random Forest Classifier
print("Training Random Forest Classifier...")
start_time = time.time()
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train_scaled, y_train)
forest_time = time.time() - start_time
print(f"Random Forest Classifier trained in {forest_time:.2f} seconds")

# Evaluate Models and print metrics for each classifier
def evaluate_classifier(clf, name, X_test, y_test):
    print(f"\n--- {name} Performance ---")

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred, accuracy

# Evaluate each classifier
sgd_pred, sgd_accuracy = evaluate_classifier(sgd_clf, "SGD Classifier", X_test_scaled, y_test)
svm_pred, svm_accuracy = evaluate_classifier(svm_clf, "SVM Classifier", X_test_scaled[:1000], y_test[:1000])
rf_pred, rf_accuracy = evaluate_classifier(forest_clf, "Random Forest Classifier", X_test_scaled, y_test)

# Confusion Matrix Visualization
plt.figure(figsize=(10, 10))
cm = confusion_matrix(y_test, sgd_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SGD Classifier")
plt.show()

# Errors
# Find examples of misclassifications
print("\n--- Error Analysis ---")
y_pred = sgd_pred  # Let's analyze errors from the SGD classifier
misclassified_indices = np.where(y_pred != y_test)[0]

# Plot some misclassified examples
if len(misclassified_indices) > 0:
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(misclassified_indices[:15]):  # Show up to 15 misclassified examples
        plt.subplot(3, 5, i + 1)
        plot_digit(X_test_array[idx])
        plt.title(f"Pred: {y_pred[idx]}, True: {y_test[idx]}")
    plt.tight_layout()
    plt.show()

# One-vs-All (OvA) Strategy
def train_binary_classifier(digit, X, y):
    """Train a binary classifier to detect if an image is a specific digit."""
    y_binary = (y == digit).astype(int)  # 1 if digit, 0 otherwise
    clf = SGDClassifier(random_state=42)
    clf.fit(X, y_binary)
    return clf

# Let's train binary classifiers for a few digits
print("\n--- One-vs-All (OvA) Strategy ---")
binary_clfs = {}
for digit in range(5):  # digits 0-4 for brevity
    binary_clfs[digit] = train_binary_classifier(digit, X_train_scaled[:5000], y_train[:5000])

# Test the binary classifiers on a sample image
sample_idx = 1234
sample_img = X_test_scaled[sample_idx]
sample_digit = y_test[sample_idx]

print(f"Sample digit: {sample_digit}")
print("Binary classifier decision scores:")
for digit, clf in binary_clfs.items():
    decision_score = clf.decision_function([sample_img])[0]
    print(f"Is digit {digit}? Score: {decision_score:.4f}")

# Model Comparison
models = ["SGD Classifier", "SVM Classifier*", "Random Forest"]
accuracies = [sgd_accuracy, svm_accuracy, rf_accuracy]
training_times = [sgd_time, svm_time, forest_time]

plt.figure(figsize=(12, 5))

# Plot accuracies
plt.subplot(1, 2, 1)
plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)

# Plot training times
plt.subplot(1, 2, 2)
plt.bar(models, training_times, color=['blue', 'green', 'orange'])
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')
plt.yscale('log')

plt.tight_layout()
plt.show()