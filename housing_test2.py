from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tar:
            housing_tar.extractall(path="datasets")
    return pd.read_csv("datasets/housing/housing.csv")

# Load data
housing = load_housing_data()

# Create income categories for stratified splitting
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# Stratified train and test split
strat_train_set, strat_test_set = train_test_split(housing,
                                                   test_size=0.2,
                                                   stratify=housing["income_cat"],
                                                   random_state=42)

# Drop the income category column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Separate features and labels
housing_train = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Define attribute lists
num_attribs = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income"
]
cat_attribs = ["ocean_proximity"]

# Create pipelines for numerical and categorical data
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# Combine pipelines with ColumnTransformer
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# Apply transformations to the training data
housing_prepared = preprocessing.fit_transform(housing_train)

# Just to inspect the transformed data shape
print("Transformed data shape:", housing_prepared.shape)

# Combine the preprocessing pipeline with the DecisionTreeRegressor model
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))

# Train the model
tree_reg.fit(housing_train, housing_labels)

# Evaluate on the training set
housing_predictions = tree_reg.predict(housing_train)
tree_rmse = mean_squared_error(housing_labels, housing_predictions)
print("Training RMSE:", tree_rmse)

# Prepare the test data
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# Transform the test data
X_test_prepared = preprocessing.transform(X_test)

# Evaluate on the test set
test_predictions = tree_reg.predict(X_test)
test_rmse = mean_squared_error(y_test, test_predictions)
print("Test RMSE:", test_rmse)
