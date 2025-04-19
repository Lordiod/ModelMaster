"""
ModelMaster - Machine Learning Classification Algorithms Module

This module contains implementations of various classification algorithms and utility 
functions for the ModelMaster application. It provides standard interfaces for training
and testing different machine learning models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import RFE

# Global variables
dataset = None
num_feature = None
model = None
X_test = None
y_test = None

#------------------------------------------{Preprocessing}------------------------------------------#

def rfe(estimator):
    """
    Performs Recursive Feature Elimination to select the best features from the dataset.
    
    Args:
        estimator: The estimator to use for feature selection
        
    Returns:
        tuple: (X, y) where X contains selected features and y contains target values
    """
    if dataset is None:
        raise ValueError("No dataset loaded. Please load a dataset first.")
        
    if num_feature is None or num_feature <= 0:
        raise ValueError("Invalid number of features specified")
        
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    selector = RFE(estimator=estimator, n_features_to_select=num_feature)
    X = selector.fit_transform(X=X, y=y)
    
    return X, y

#------------------------------------------{Algorithm Implementations}------------------------------#

def linear_regression(test_size):
    """
    Trains a Linear Regression model on the dataset.
    
    Args:
        test_size: Proportion of the dataset to be used as test set
    """
    global model, X_test, y_test

    X, y = rfe(LinearRegression())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = LinearRegression()
    model = clf.fit(X_train, y_train)

    tkinter.messagebox.showinfo("Success", "Linear Regression model trained successfully")

def logistic_regression(test_size):
    """
    Trains a Logistic Regression model on the dataset.
    
    Args:
        test_size: Proportion of the dataset to be used as test set
    """
    global model, X_test, y_test

    X, y = rfe(LogisticRegression())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = LogisticRegression()
    model = clf.fit(X_train, y_train)

    tkinter.messagebox.showinfo("Success", "Logistic Regression model trained successfully")

def knn(n_neighbors, test_size):
    """
    Trains a K-Nearest Neighbors model on the dataset.
    
    Args:
        n_neighbors: Number of neighbors to use
        test_size: Proportion of the dataset to be used as test set
    """
    global model, X_test, y_test

    X, y = rfe(KNeighborsClassifier(n_neighbors=n_neighbors))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    model = clf.fit(X_train, y_train)

    tkinter.messagebox.showinfo("Success", "KNN model trained successfully")

def decision_tree(max_depth, test_size):
    """
    Trains a Decision Tree model on the dataset.
    
    Args:
        max_depth: Maximum depth of the tree
        test_size: Proportion of the dataset to be used as test set
    """
    global model, X_test, y_test

    X, y = rfe(DecisionTreeClassifier(max_depth=max_depth))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = DecisionTreeClassifier(max_depth=max_depth)
    model = clf.fit(X_train, y_train)

    tkinter.messagebox.showinfo("Success", "Decision Tree model trained successfully")

def SVM(kernel, test_size):
    """
    Trains a Support Vector Machine model on the dataset.
    
    Args:
        kernel: Kernel type to be used in the algorithm
        test_size: Proportion of the dataset to be used as test set
    """
    global model, X_test, y_test
    
    # Using linear kernel for RFE since rbf doesn't support coef_ or feature_importances_
    X, y = rfe(SVC(kernel="linear"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = SVC(kernel=kernel)
    model = clf.fit(X_train, y_train)

    tkinter.messagebox.showinfo("Success", f"SVM model with {kernel} kernel trained successfully")

#------------------------------------------{Evaluation Methods}------------------------------------#

def display_confusion_matrix(conf_matrix):
    """
    Displays the confusion matrix using matplotlib.
    
    Args:
        conf_matrix: Confusion matrix to display
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.matshow(conf_matrix, cmap='Blues')

    # Add text annotations
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, str(conf_matrix[i, j]), va='center', ha='center', fontsize=12)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.show()

def tester():
    """
    Tests the trained model and calculates performance metrics.
    
    Returns:
        bool: True if testing was successful, False otherwise
    """
    if 'model' not in globals() or model is None:
        tkinter.messagebox.showerror("Error", "Please train a model first")
        return False
        
    if X_test is None or y_test is None:
        tkinter.messagebox.showerror("Error", "Test data not found. Please train a model first")
        return False
        
    global conf_matrix, accuracy, precision, recall, f1_score

    try:
        prediction = model.predict(X_test)
        
        # Check if the output needs to be converted to binary values (for regression models)
        if isinstance(model, LinearRegression):
            y_pred = [1 if y >= 0.5 else 0 for y in prediction]
        else:
            y_pred = prediction
            
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # Avoid division by zero
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0

        tkinter.messagebox.showinfo("Success", "Model tested successfully")
        return True
        
    except Exception as e:
        tkinter.messagebox.showerror("Error", f"Testing failed: {str(e)}")
        return False