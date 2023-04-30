import numpy as np
from sklearn.linear_model import LinearRegression

# Create some example training data
x_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

# Create a linear regression model and train it on the training data
model = LinearRegression()
model.fit(x_train, y_train)

# Make a prediction for a new input value
x_new = np.array([[8]])
y_pred = model.predict(x_new)

# Print the predicted output
print(y_pred)
