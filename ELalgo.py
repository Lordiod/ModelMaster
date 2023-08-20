import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score , precision_score , recall_score
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import tkinter 

# global variables
dataset = None
num_feature = None

#------------------------------------------{EL Preprocessing}------------------------------------------#

# method {Recursive Feature Elimination} to get the reduced X and Y

def rfe(estimator):
   X = dataset.iloc[:, :-1].values
   y = dataset.iloc[:, -1].values
   selector = RFE(estimator = estimator, n_features_to_select= num_feature)
   X = selector.fit_transform(X=X, y=y)
   return X, y

#------------------------------------------{EL Algorithms}---------------------------------------------#

#def linear_regression(test_size):
#  global model, X_test, y_test
#
#  X, y = rfe(LinearRegression())
#  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#
#  sc = StandardScaler()
#  X_train = sc.fit_transform(X_train)
#  X_test = sc.transform(X_test)
#
#  clf = LinearRegression()
#  model = clf.fit(X_train, y_train)
#
#  tkinter.messagebox.showinfo("Successful","Model Trained Successfully")
#  return

def logistic_regression(test_size):
   global model, X_test, y_test

   X, y = rfe(LogisticRegression())
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)

   clf = LogisticRegression()
   model = clf.fit(X_train, y_train)

   tkinter.messagebox.showinfo("Successful", "Model Trained Successfully")
   return

def knn(n_neighbors, test_size):
   global model, X_test, y_test

   X, y = rfe(KNeighborsClassifier(n_neighbors=n_neighbors))
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)

   clf = KNeighborsClassifier(n_neighbors=n_neighbors)
   model = clf.fit(X_train, y_train)

   tkinter.messagebox.showinfo("Successful", "Model Trained Successfully")
   return

def decision_tree(max_depth ,test_size):
  global model, X_test, y_test

  X, y = rfe(DecisionTreeClassifier(max_depth=max_depth))
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  clf = DecisionTreeClassifier(max_depth=max_depth)
  model = clf.fit(X_train, y_train)

  tkinter.messagebox.showinfo("Successful","Model Trained Successfully")
  return

def SVM(kernel, test_size):
  global model, X_test, y_test
  #estimator = SVR(kernel="linear") linear becuase rbf does't support coef_ nor feature_importances_
  X, y = rfe(SVC(kernel = "linear"))

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  clf = SVC(kernel = kernel)
  model = clf.fit(X_train, y_train)

  tkinter.messagebox.showinfo("Successful","Model Trained Successfully")
  return

#------------------------------------------{EL Testing}---------------------------------------------#

def display_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.matshow(conf_matrix, cmap='Blues')

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, str(conf_matrix[i, j]), va='center', ha='center')

    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()


def tester():
   try:
      model
   except NameError:
      tkinter.messagebox.showerror("Error","Please Train the model first")  
   else:
    global conf_matrix, accuracy, precision, recall, f1_score

    prediction = model.predict(X_test)
    y_pred = [1 if y >= 0.5 else 0 for y in prediction] 

    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy= accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = 2 * (precision * recall) / (precision + recall)

    tkinter.messagebox.showinfo("Successful","Model Tested Successfully")
    return   

#def test_ready_model():
#   global X_test , y_test
#   sc = StandardScaler()
#   X_test=sc.fit_transform(dataset.iloc[:, :-1].values)
#   y_test=dataset.iloc[:, -1].values
#   tester()