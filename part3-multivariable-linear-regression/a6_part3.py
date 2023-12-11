import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values

#split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)

#create linear regression model
model = LinearRegression().fit(xtrain, ytrain)


#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y),2)
print(f"Model's Linear Equation: y={coef[0]}x1 + {coef[1]}x2 + {intercept}")
print("R Squared value:", r_squared)
print("***************")
print("Testing Results")



#Loop through the data and print out the predicted prices and the 
#actual prices
predictions = model.predict(xtest)
predict=np.around(predictions , 2)
print(predict)

for index in range(len(xtest)):
    actual = round(ytest[index])
    predicted_y = predict[index] 
    x_coord = xtest[index]
    x_coord=np.around(x_coord,2)
    
    print(f"miles: {x_coord[0]} age: {x_coord[1]} Actual: {actual} Predicted: {predicted_y}")
print(f"miles: {x_coord[89]} age: {x_coord[10]} Actual: {actual} Predicted: {predicted_y}")
print(f"miles: {x_coord[150]} age: {x_coord[20]} Actual: {actual} Predicted: {predicted_y}")