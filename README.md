# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters: Start by initializing the parameters (weights) theta with random values or zeros.
2. Compute Sigmoid Function: Define the sigmoid function that maps any real-valued number to a value between 0 and 1.
3. Compute Loss Function: Define the loss function, which measures the error between the predicted output and the actual output.
4. Gradient Descent Optimization: Implement the gradient descent algorithm to minimize the loss function. In each iteration, compute the gradient of the loss function with respect to the parameters (theta), and update the parameters in the opposite direction of the gradient to minimize the loss.
5. Iterate Until Convergence: Repeat the gradient descent steps for a predefined number of iterations or until convergence criteria are met. Convergence can be determined when the change in the loss function between iterations becomes very small or when the parameters (theta) stop changing significantly.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Rakshitha J
RegisterNumber: 212223240135
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y = Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
## Output:
### Dataset
![image](https://github.com/user-attachments/assets/4b2f81b0-232d-4c9a-ba69-df0891ea857e)


### Data Types
![image](https://github.com/user-attachments/assets/0f3cdced-f918-4afb-ba32-982574e1a0eb)


### New Dataset
![image](https://github.com/user-attachments/assets/b6f68b14-aa45-4284-a040-6c257cdf17c4)


### Y values
![image](https://github.com/user-attachments/assets/5220bb05-d46c-43ab-b883-03ef68ec4abc)


### Accuracy
![image](https://github.com/user-attachments/assets/689ac8d7-b737-439c-93e8-5f69af371e09)


### Y pred
![image](https://github.com/user-attachments/assets/8faf59c9-48cc-4041-8925-333534bc7f0e)


### New Y 
![image](https://github.com/user-attachments/assets/158962a6-0c90-4a0e-a852-6f11ba202f13)


![image](https://github.com/user-attachments/assets/89300624-71cf-4a10-9958-af0993e9ed15)


![image](https://github.com/user-attachments/assets/d542c7fe-40a7-4a94-bb31-299004eb2ba2)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

