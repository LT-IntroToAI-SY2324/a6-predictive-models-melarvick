import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
print(data)
x = data["Age"].values
y = data["Blood Pressure"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)

# Create the model
model = LinearRegression().fit(x, y)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)
print(coef, intercept, r_squared)

# Print out the linear equation and r squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")
print(f"Prediction when x is {x_predict}: {prediction}")


# Predict the the blood pressure of someone who is 43 years old.
x_predict = 43
# plug that value into your model
prediction = model.predict([[x_predict]])
print(x_predict)


# Create the model in matplotlib and include the line of best fit
# sets the size of the graph
plt.figure(figsize=(6, 4))

# creates a scatter plot and labels the axes
plt.scatter(x,y)
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure by Age")

# show the plot
plt.show()