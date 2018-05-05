# Delayed Flight Predictor
# pip install --ignore-installed six tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

def encodeOutputVariable(y):
    labelencoder_Y_Origin = LabelEncoder()
    y = labelencoder_Y_Origin.fit_transform(y.astype(str))
    return y

def encodeCategoricalData(X, index):
    # encode categorical data
    labelencoder_X_Origin = LabelEncoder()
    X[:, index] = labelencoder_X_Origin.fit_transform(X[:, index].astype(str))
    return X    

def encodeHotEncoder(X, numberOfCategories):
    onehotencoder = OneHotEncoder(categorical_features = [numberOfCategories])
    X = onehotencoder.fit_transform(X.astype(str)).toarray()    
    X = X[:, 1:]
    return X

def minimumValues(train):
    return [0 if math.isnan(x) else x for x in train]

def minimumDelay(x):
    return 0 if np.isnan(x) or x < 0 else x


# importing the data
dataset = pd.read_csv("./data/employee_attrition.csv", nrows = 200)
X = dataset.iloc[:, 2:34].values    
y = dataset.iloc[:, 1].values

# encode categorical data
X = encodeCategoricalData(X, 0)
X = encodeCategoricalData(X, 2)
X = encodeCategoricalData(X, 5)
X = encodeCategoricalData(X, 9)
X = encodeCategoricalData(X, 13)
X = encodeCategoricalData(X, 15)
X = encodeCategoricalData(X, 19)
X = encodeCategoricalData(X, 20)

X = encodeHotEncoder(X, 8)
y = encodeOutputVariable(y)


# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# feature scaling 
sc = StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# part 2 - now let's make the ANN

# initializing the Multi Layer Perceptron 
classifier = MLPClassifier(solver="adam", alpha=1e-5, max_iter=500,
                           hidden_layer_sizes=(13, 13, 13))

# fitting the Multi Layer Perceptron to the training set
classifier.fit(X_train, y_train)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# use the threshold of error to determine whether a prediction is valid
y_pred = (y_pred > 0.5)

# making the confusion matrix
cm = confusion_matrix(y_test.ravel(), y_pred.ravel())

print(pd.crosstab(y_test.ravel(), y_pred.ravel(), rownames=['True'], colnames=['Predicted'], margins=True))
print(classification_report(y_test, y_pred))

