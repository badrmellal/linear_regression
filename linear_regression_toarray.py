from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # input with reshape because it's require for sklearn
y = np.array([36, 17, 1, 9, 11, 15, 18, 23, 25, 30])  # output

# 3-fold cross-validation (splits data into 3 parts)
kf = KFold(n_splits=3, shuffle=True, random_state=42)
model_cv = LinearRegression()

# R2 scores and MSE for each fold
cv_r2_scores = cross_val_score(model_cv, x, y, cv=kf, scoring='r2')
cv_mse_scores = -cross_val_score(model_cv, x, y, cv=kf, scoring='neg_mean_squared_error')

# cross validation results
print("\nCross-Validation Results:")
print(f"R2 scores for each fold: {cv_r2_scores}")
print(f"Mean R2 score: {np.mean(cv_r2_scores):.4f}")
print(f"MSE scores for each fold: {cv_mse_scores}")
print(f"Mean MSE: {np.mean(cv_mse_scores):.4f}")


# Splitting the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

# let's do predictions on both training and test sets
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# next we evaluate the model on training data
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2_score = r2_score(y_train, y_train_pred)
print(f"Training Mean Squared Error: {train_mse}")
print(f"Training R2 Score: {train_r2_score}")

# next on test data
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test Mean Squared Error: {test_mse}")
print(f"Test R2 Score: {test_r2}")

# finally we predict for new data
new_x = np.array([12, 15, 20]).reshape(-1, 1)
new_y_pred = model.predict(new_x)
print(f"Predictions for new x values: {new_y_pred}")

# let's see the results
plt.figure(figsize=(12, 8))

# plotting
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.scatter(x_train, y_train_pred, color='lightblue', marker='x', label='Training predictions')

# test data and predictions
plt.scatter(x_test, y_test, color='red', label='Test data')
plt.scatter(x_test, y_test_pred, color='lightcoral', marker='x', label='Test predictions')

# regression line
x_range = np.linspace(min(x), max(new_x), 100).reshape(-1, 1)
y_range_pred = model.predict(x_range)
plt.plot(x_range, y_range_pred, color='green', label='Regression line')

# plotting new predictions
plt.scatter(new_x, new_y_pred, color='purple', marker='d', s=100, label='New predictions')

plt.title('Linear regression with a simple array')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
