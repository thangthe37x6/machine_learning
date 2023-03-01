from lr import linear
import matplotlib.pyplot as plt
import numpy as np

# Generate a dataset of 50 points with some noise
numofpoint = 50
noise = np.random.normal(0,1, numofpoint).reshape(-1,1)
x_train = np.linspace(20,200,numofpoint).reshape(-1,1)
N = x_train.shape[0]
y_train = 15*x_train + 8 + 20*noise

# Plot the training data
plt.scatter(x_train,y_train)

# Fit a linear regression model to the training data
lr = linear()
lr.fit(x_train,y_train)
print(f"bias:{lr.bias} weight:{lr.w}")

# Plot the predicted values from the model
plt.scatter(x_train, y_train)  
y_predicted = lr.predicted(x_train) 
plt.plot(x_train, y_predicted, 'b')  
plt.show()