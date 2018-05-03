# Delayed Flight Predictor
# pip install --ignore-installed six tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

def encodeCategoricalData(X, index):
    # encode categorical data
    labelencoder_X_Origin = LabelEncoder()
    X[:, index] = labelencoder_X_Origin.fit_transform(X[:, index].astype(str))
    return X    

def encodeHotEncoder(X):
    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()    
    X = X[:, 1:]
    return X

def minimumDelay(x):
    return 0 if np.isnan(x) or x < 0 else x


# importing the data
dataset = pd.read_csv("./data/flights.csv", nrows = 1000)
minimalFunc = np.vectorize(minimumDelay)
X = dataset.iloc[:, 7:10].values        
y = minimalFunc(dataset.iloc[:, 11].values)

# encode categorical data
#X = encodeCategoricalData(X, 4)
#X = encodeCategoricalData(X, 6)
#X = encodeCategoricalData(X, 7)
#X = encodeCategoricalData(X, 8)

X = encodeCategoricalData(X, 0)
X = encodeCategoricalData(X, 1)


X = encodeHotEncoder(X)


# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# part 2 - now let's make the ANN

# initializing the Multi Layer Perceptron 
classifier = MLPClassifier(solver="lbfgs", alpha=1e-5, 
                           hidden_layer_sizes=(5, 3), random_state=1)

# fitting the Multi Layer Perceptron to the training set
classifier.fit(X_train, y_train)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# use the threshold of error to determine whether a prediction is valid
y_pred = (y_pred > 0.5)

# making the confusion matrix
cm = confusion_matrix(y_test.ravel(), y_pred.ravel())

#pd.crosstab(y_test.ravel(), y_pred.ravel(), rownames=['True'], colnames=['Predicted'], margins=True)

