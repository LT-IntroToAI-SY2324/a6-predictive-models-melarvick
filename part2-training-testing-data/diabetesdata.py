import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True)
y= data.target 
data= data.frame
print(data)
x= data["bmi"]
print(x)
print(y)