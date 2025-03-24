import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# download and prepare data
data_root = "https://raw.githubusercontent.com/ageron/data/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
Y = lifesat["Life satisfaction"].values

# visualize the data
lifesat.plot(kind="scatter", grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# select a linear regression model
model = LinearRegression()

# train the model
model.fit(X, Y)

# make a prediction
X_new = [[37_655.2]] # GDP of Cyprus which is not in the dataset

print(model.predict(X_new))

morocco_gdp_per_capita = [[3_672]]  # GDP per capita of Morocco
predicted_life_satisfaction = model.predict(morocco_gdp_per_capita)

print(predicted_life_satisfaction)