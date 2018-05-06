# HR Employee Attrition Predictor
# pip install --ignore-installed six tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
import sklearn.metrics as metrics

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap


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

def outputPredictorResults(y_test, y_pred, title):
    # output results for Naive Bayes Classification
    print("For", title, "Classification")
    print(metrics.accuracy_score(y_test, y_pred) * 100)
    print(metrics.confusion_matrix(y_test, y_pred))
    print(pd.crosstab(y_test.ravel(), y_pred.ravel(), rownames=['True'], colnames=['Predicted'], margins=True))
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.zero_one_loss(y_test, y_pred))
    print(metrics.log_loss(y_test, y_pred))
    print(metrics.roc_auc_score(y_test, y_pred))

def renderDecisionTreeGraph(dot_data):
    graph = graphviz.Source(dot_data)
    graph.render("EmployeeAttritionDecisionTree")
    graph

# developing the Naive Bayes model    
def creatingBayesPredictor(X_train, y_train, X_test, y_test):
    # utilize the Gaussian Naive Bayes algorithm for classification
    bayes_classifier = GaussianNB()
    
    # fitting the Gaussian Naive Bayes with the training data
    bayes_classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    bayes_y_pred = bayes_classifier.predict(X_test)
    
    # use the threshold of error to determine whether a prediction is valid
    bayes_y_pred = (bayes_y_pred > 0.5)
    
    # make the confusion matrix
    cm = confusion_matrix(y_test, bayes_y_pred)
    
    #output results
    outputPredictorResults(y_test, bayes_y_pred, "Naive Bayes")

# developing the Decision Tree model    
def creatingDecisionTreePredictor(X_train, y_train, X_test, y_test):
    # initializing the Decision Tree Classification
    decision_tree_classifier = tree.DecisionTreeClassifier()
    
    # fitting the Decision Tree Model to the training set
    decision_tree_classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    decision_y_pred = decision_tree_classifier.predict(X_test)
    
    # use the threshold of error to determine whether a prediction is valid
    decision_y_pred = (decision_y_pred > 0.5)
    
    # graph the tree with the various conditions and samples info
    dot_data = tree.export_graphviz(decision_tree_classifier, 
                                    out_file = None,
                                    feature_names = dataset.columns.values,
                                    filled=True, 
                                    rounded=True,
                                    impurity=False,
                                    special_characters=True)
    
    # output results and graph
    outputPredictorResults(y_test, decision_y_pred, "Decision Tree")
    renderDecisionTreeGraph(dot_data)    

# developing the Multi Layer Perceptron Neural Network
def creatingNeuralNetworkPredictor(X_train, y_train, X_test, y_test):
    # initialize the Multi Layer Perceptron Neural Network 
    mlp_classifier = MLPClassifier(solver="adam", alpha=1e-5, max_iter=500,
                               hidden_layer_sizes=(13, 13, 13))
    
    # fitting the Multi Layer Perceptron to the training set
    mlp_classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    mlp_y_pred = mlp_classifier.predict(X_test)
    
    # use the threshold of error to determine whether a prediction is valid
    mlp_y_pred = (mlp_y_pred > 0.5)
    
    # making the confusion matrix
    cm = confusion_matrix(y_test.ravel(), mlp_y_pred.ravel())
    
    # output results
    outputPredictorResults(y_test, mlp_y_pred, "Neural Network")

# importing the data
dataset = pd.read_csv("./data/employee_attrition.csv")
X = dataset.iloc[:, 2:34].values    
y = dataset.iloc[:, 1].values
models = {"bayes": {}, "neural": {}, "decision": {}}

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

X_train_filter = sc.fit_transform(X_train_filter)
X_test_filter = sc.transform(X_test_filter)

# outputting data summary
print("Summary Info About the Dataset")
print("Does category contain null values?")
print(dataset.isnull().any(), "\n")
print("Said Yes to Attrition: ", y[(y == 1)].size)
print("Said No to Attrition: ", y[(y == 0)].size)
print("Total responses: ", y.size)

creatingBayesPredictor(X_train, y_train, X_test, y_test)
creatingDecisionTreePredictor(X_train, y_train, X_test, y_test)
creatingNeuralNetworkPredictor(X_train, y_train, X_test, y_test)
