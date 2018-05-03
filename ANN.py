# Delayed Flight Predictor
# pip install --ignore-installed six tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# importing the keras libraries and packages
import keras
from keras.models import Sequential 
from keras.layers import Dense 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def encodeCategoricalData(X, index):
    # encode categorical data
    labelencoder_X_Origin = LabelEncoder()
    X[:, index] = labelencoder_X_Origin.fit_transform(X[:, index].astype(str))
    return X    

def encodeHotEncoder(X, index):
    onehotencoder = OneHotEncoder(categorical_features = [index])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    return X

def minimumValues(train):
    return [0 if math.isnan(x) else x for x in train]

def minimumDelay(x):
    return 0 if np.isnan(x) or x < 0 else x


# importing the data
dataset = pd.read_csv("./data/echocardiogram.csv")
minimalFunc = np.vectorize(minimumDelay)
#X = dataset.iloc[:, 4:10].values        
X = np.apply_along_axis(minimumValues, axis=1, arr=dataset.iloc[:, 2:9].values)        
y = minimalFunc(dataset.iloc[:, 1].values)

# encode categorical data
#==============================================================================
# X = encodeCategoricalData(X, 0)
# X = encodeCategoricalData(X, 2)
# X = encodeCategoricalData(X, 3)
# X = encodeCategoricalData(X, 4)
#==============================================================================

#==============================================================================
# X = encodeCategoricalData(X, 0)
# X = encodeCategoricalData(X, 1)
# 
# 
# X = encodeHotEncoder(X, 1)
# X = encodeHotEncoder(X, 1)
#==============================================================================


# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# part 2 - now let's make the ANN

# initializing the ANN 
classifier = Sequential()

# adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=7, units=6, kernel_initializer="uniform"))

# adding the second hidden layer 
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# adding the output layer 
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# use the threshold of error to determine whether a prediction is valid
y_pred = (y_pred > 0.5)

# making the confusion matrix
cm = confusion_matrix(y_test.ravel(), y_pred.ravel())

pd.crosstab(y_test.ravel(), y_pred.ravel(), rownames=['True'], colnames=['Predicted'], margins=True)

y_pred_values = classifier.predict_classes(X_test)

