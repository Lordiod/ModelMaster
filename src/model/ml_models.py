"""
ModelMaster - Machine Learning Classification Model Component

This module contains implementations of various classification algorithms and utility 
functions for the ModelMaster application. It provides standard interfaces for training
and testing different machine learning models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE


class MLModel:
    """Base class for all machine learning models in the application."""
    
    def __init__(self):
        """Initialize MLModel with default values."""
        self.dataset = None
        self.num_feature = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "conf_matrix": None
        }
    
    def load_dataset(self, dataset_path):
        """Load dataset from a CSV file.
        
        Args:
            dataset_path: Path to the CSV file
            
        Returns:
            dict: {"success": bool, "message": str, "features": int, "rows": int} if successful, 
                  {"success": False, "message": str} if failed
        """
        try:
            self.dataset = pd.read_csv(dataset_path)
            num_features = self.dataset.shape[1] - 1  # Assuming last column is the target
            num_rows = self.dataset.shape[0]
            return {
                "success": True,
                "message": "Dataset loaded successfully",
                "features": num_features,
                "rows": num_rows
            }
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return {
                "success": False,
                "message": f"Error loading dataset: {str(e)}"
            }
    
    def rfe(self, estimator):
        """Performs Recursive Feature Elimination to select the best features from the dataset.
        
        Args:
            estimator: The estimator to use for feature selection
            
        Returns:
            tuple: (X, y) where X contains selected features and y contains target values
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please load a dataset first.")
            
        if self.num_feature is None or self.num_feature <= 0:
            raise ValueError("Invalid number of features specified")
            
        X = self.dataset.iloc[:, :-1].values
        y = self.dataset.iloc[:, -1].values
        
        selector = RFE(estimator=estimator, n_features_to_select=self.num_feature)
        X = selector.fit_transform(X=X, y=y)
        
        return X, y
    
    def train_model(self, test_size, **kwargs):
        """Base method for training a model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement train_model method")
    
    def test_model(self):
        """Tests the trained model and calculates performance metrics.
        
        Returns:
            bool: True if testing was successful, False otherwise
        """
        if self.model is None:
            print("Error: Please train a model first")
            return False
            
        if self.X_test is None or self.y_test is None:
            print("Error: Test data not found. Please train a model first")
            return False
            
        try:
            prediction = self.model.predict(self.X_test)
            
            # Check if the output needs to be converted to binary values (for regression models)
            if isinstance(self.model, LinearRegression):
                y_pred = [1 if y >= 0.5 else 0 for y in prediction]
            else:
                y_pred = prediction
                
            self.metrics["conf_matrix"] = confusion_matrix(self.y_test, y_pred)
            self.metrics["accuracy"] = accuracy_score(self.y_test, y_pred)
            self.metrics["precision"] = precision_score(self.y_test, y_pred, zero_division=0)
            self.metrics["recall"] = recall_score(self.y_test, y_pred, zero_division=0)
            
            # Calculate F1 score
            self.metrics["f1_score"] = f1_score(self.y_test, y_pred, zero_division=0)

            return True
            
        except Exception as e:
            print(f"Testing failed: {str(e)}")
            return False
    
    def display_confusion_matrix(self):
        """Displays the confusion matrix using matplotlib."""
        if self.metrics["conf_matrix"] is None:
            print("Error: No confusion matrix available. Please test the model first.")
            return
            
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.matshow(self.metrics["conf_matrix"], cmap='Blues')

        # Add text annotations
        for i in range(self.metrics["conf_matrix"].shape[0]):
            for j in range(self.metrics["conf_matrix"].shape[1]):
                ax.text(j, i, str(self.metrics["conf_matrix"][i, j]), va='center', ha='center', fontsize=12)

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.show()


class LinearRegressionModel(MLModel):
    """Linear Regression model implementation."""
    
    def train_model(self, test_size, **kwargs):
        """Trains a Linear Regression model on the dataset.
        
        Args:
            test_size: Proportion of the dataset to be used as test set
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            X, y = self.rfe(LinearRegression())
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            self.X_test = X_test
            self.y_test = y_test
            return True
        except Exception as e:
            print(f"Error training Linear Regression model: {str(e)}")
            return False


class LogisticRegressionModel(MLModel):
    """Logistic Regression model implementation."""
    
    def train_model(self, test_size, **kwargs):
        """Trains a Logistic Regression model on the dataset.
        
        Args:
            test_size: Proportion of the dataset to be used as test set
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            X, y = self.rfe(LogisticRegression())
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            self.model = LogisticRegression()
            self.model.fit(X_train, y_train)
            self.X_test = X_test
            self.y_test = y_test
            return True
        except Exception as e:
            print(f"Error training Logistic Regression model: {str(e)}")
            return False


class KNNModel(MLModel):
    """K-Nearest Neighbors model implementation."""
    
    def train_model(self, test_size, **kwargs):
        """Trains a K-Nearest Neighbors model on the dataset.
        
        Args:
            test_size: Proportion of the dataset to be used as test set
            n_neighbors: Number of neighbors to use (from kwargs)
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if 'n_neighbors' not in kwargs:
            print("Error: n_neighbors parameter is required for KNN model")
            return False
            
        try:
            n_neighbors = kwargs['n_neighbors']
            X, y = self.rfe(KNeighborsClassifier(n_neighbors=n_neighbors))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.model.fit(X_train, y_train)
            self.X_test = X_test
            self.y_test = y_test
            return True
        except Exception as e:
            print(f"Error training KNN model: {str(e)}")
            return False


class DecisionTreeModel(MLModel):
    """Decision Tree model implementation."""
    
    def train_model(self, test_size, **kwargs):
        """Trains a Decision Tree model on the dataset.
        
        Args:
            test_size: Proportion of the dataset to be used as test set
            max_depth: Maximum depth of the tree (from kwargs)
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if 'max_depth' not in kwargs:
            print("Error: max_depth parameter is required for Decision Tree model")
            return False
            
        try:
            max_depth = kwargs['max_depth']
            X, y = self.rfe(DecisionTreeClassifier(max_depth=max_depth))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            self.model = DecisionTreeClassifier(max_depth=max_depth)
            self.model.fit(X_train, y_train)
            self.X_test = X_test
            self.y_test = y_test
            return True
        except Exception as e:
            print(f"Error training Decision Tree model: {str(e)}")
            return False


class SVMModel(MLModel):
    """Support Vector Machine model implementation."""
    
    def train_model(self, test_size, **kwargs):
        """Trains a Support Vector Machine model on the dataset.
        
        Args:
            test_size: Proportion of the dataset to be used as test set
            kernel: Kernel type to be used in the algorithm (from kwargs)
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if 'kernel' not in kwargs:
            print("Error: kernel parameter is required for SVM model")
            return False
            
        try:
            kernel = kwargs['kernel']
            # Using linear kernel for RFE since rbf doesn't support coef_ or feature_importances_
            X, y = self.rfe(SVC(kernel="linear"))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            self.model = SVC(kernel=kernel)
            self.model.fit(X_train, y_train)
            self.X_test = X_test
            self.y_test = y_test
            return True
        except Exception as e:
            print(f"Error training SVM model: {str(e)}")
            return False
