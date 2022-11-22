# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

## Neural Network Model
![model](https://user-images.githubusercontent.com/114344373/192255825-adf9a1ff-501e-475a-bc38-e40b22b4621b.jpg)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```sh
Developed by: NITISH KUMAR R
Registration Number:212219220036

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
data1 = pd.read_csv('Book1new.csv')
data1.head()
X = data1[['input']].values
X
Y = data1[["output"]].values
Y
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scalar=MinMaxScaler()
scalar.fit(X_train)
scalar.fit(X_test)
X_train=scalar.transform(X_train)
X_test=scalar.transform(X_test)
import tensorflow as tf
model=tf.keras.Sequential([tf.keras.layers.Dense(4,activation='relu'),
                          tf.keras.layers.Dense(4,activation='relu'),
                          tf.keras.layers.Dense(1)])
model.compile(loss="mae",optimizer="rmsprop",metrics=["mse"])
history=model.fit(X_train,Y_train,epochs=1000)
import numpy as np
X_test
preds=model.predict(X_test)
np.round(preds)
tf.round(model.predict([[20]]))
pd.DataFrame(history.history).plot()
r=tf.keras.metrics.RootMeanSquaredError()
r(Y_test,preds)
```
## Dataset Information
![image](https://user-images.githubusercontent.com/75235813/189538394-40eec53b-a6aa-4ea2-bc62-0631958cb7b1.png)



## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235813/189538471-bd92a874-ba80-4714-b005-b7de74d1c5bf.png)

### Classification Report

![image](https://user-images.githubusercontent.com/75235813/189538508-c3918f90-7041-4324-966f-1becca9ee18d.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/75235813/189538524-e1fac22d-152c-4d15-8df3-7d4ed3976b98.png)


### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75235813/189538551-ba7ad872-cbfc-4b8d-9c25-474b6c2e456c.png)

## RESULT
Thus,the neural network classification model for the given dataset is developed.
