"""
ModelMaster - Controller Component

This module serves as a bridge between the user interface (View) and data processing (Model).
It handles the logic for responding to user actions and updating the model and view accordingly.
"""

# Standard library imports - needed for file operations
import os.path  # More specific import just for path operations
import pickle

# Local application imports - import specific classes needed
from src.model.ml_models import (
    LinearRegressionModel, 
    LogisticRegressionModel, 
    KNNModel, 
    DecisionTreeModel, 
    SVMModel
)


class ModelController:
    """Controller class to manage interactions between the GUI and models."""
    def __init__(self):
        """Initialize ModelController with default values."""
        self.current_model = None
        self.model_registry = {
            "linear_reg": LinearRegressionModel,
            "logistic_reg": LogisticRegressionModel,
            "knn": KNNModel,
            "tree": DecisionTreeModel,
            "svm": SVMModel
        }
        
    def load_dataset(self, dataset_path):
        """
        Loads a dataset from the given path or file-like object.

        Args:
            dataset_path: Path to the CSV dataset file, or a file-like object
                (e.g. a Streamlit UploadedFile) exposing a `.name` attribute

        Returns:
            dict: {"success": bool, "message": str, "filename": str, "features": int, "rows": int} or
                  {"success": False, "message": str, "filename": None}
        """
        is_file_like = hasattr(dataset_path, 'read')

        if is_file_like:
            filename = getattr(dataset_path, 'name', 'dataset.csv')
        else:
            if not os.path.exists(dataset_path):
                return {"success": False, "message": "File does not exist", "filename": None}
            filename = os.path.basename(dataset_path)

        if not filename.endswith('.csv'):
            return {"success": False, "message": "File must be a CSV", "filename": None}

        # Create a generic model to load the dataset
        self.current_model = self.model_registry["linear_reg"]()

        result = self.current_model.load_dataset(dataset_path)
        if result["success"]:
            filename = filename if is_file_like else os.path.basename(dataset_path)
            return {
                "success": True, 
                "message": f"Dataset '{filename}' loaded successfully!\nFeatures: {result['features']}\nRows: {result['rows']}", 
                "filename": filename,
                "features": result["features"],
                "rows": result["rows"]
            }
        else:
            return {"success": False, "message": result["message"], "filename": None}
            
    def set_feature_count(self, num_features):
        """
        Sets the number of features to use in the model.
        
        Args:
            num_features: Number of features to use
            
        Returns:
            dict: {"success": bool, "message": str}
        """
        if self.current_model is None:
            return {"success": False, "message": "No dataset loaded"}
            
        try:
            num_features = int(num_features)
            max_features = self.current_model.dataset.shape[1] - 1  # All columns except target
            
            if num_features <= 0:
                return {"success": False, "message": "Number of features must be positive"}
                
            if num_features > max_features:
                return {
                    "success": False, 
                    "message": f"Number of features ({num_features}) cannot exceed the available features ({max_features})!"
                }
                
            self.current_model.num_feature = num_features
            return {"success": True, "message": f"Feature count set to {num_features}"}
        except ValueError:
            return {"success": False, "message": "Invalid number format"}
        except Exception as e:
            return {"success": False, "message": f"Error setting feature count: {str(e)}"}
    
    def select_algorithm(self, algorithm_code):
        """
        Selects and instantiates the appropriate algorithm class.
        
        Args:
            algorithm_code: String identifier for the algorithm
            
        Returns:
            bool: True if successful, False otherwise
        """
        if algorithm_code not in self.model_registry:
            return False
            
        dataset = None
        num_feature = None
        
        # Preserve dataset and feature count if a model was already created
        if self.current_model:
            dataset = self.current_model.dataset
            num_feature = self.current_model.num_feature
        
        # Create new model instance
        self.current_model = self.model_registry[algorithm_code]()
        
        # Restore dataset and feature count
        if dataset is not None:
            self.current_model.dataset = dataset
        if num_feature is not None:
            self.current_model.num_feature = num_feature
            
        return True
    
    def train_model(self, algorithm_code, test_size, **kwargs):
        """
        Trains the model with the given parameters.
        
        Args:
            algorithm_code: String identifier for the algorithm
            test_size: Proportion of the dataset to be used as test set
            **kwargs: Additional parameters for specific algorithms
            
        Returns:
            dict: {"success": bool, "message": str}
        """
        if self.current_model is None:
            return {"success": False, "message": "No model selected"}
            
        try:
            success = self.current_model.train_model(float(test_size), **kwargs)
            
            if success:
                return {"success": True, "message": f"{algorithm_code.replace('_', ' ').title()} model trained successfully"}
            else:
                return {"success": False, "message": "Model training failed"}
                
        except Exception as e:
            return {"success": False, "message": f"Error during training: {str(e)}"}
    
    def test_model(self):
        """
        Tests the trained model with the test dataset.
        
        Returns:
            dict: {"success": bool, "message": str}
        """
        if self.current_model is None:
            return {"success": False, "message": "No model selected"}
            
        if self.current_model.model is None:
            return {"success": False, "message": "Model not trained"}
            
        try:
            success = self.current_model.test_model()
            
            if success:
                return {"success": True, "message": "Model tested successfully"}
            else:
                return {"success": False, "message": "Model testing failed"}
                
        except Exception as e:
            return {"success": False, "message": f"Error during testing: {str(e)}"}
    
    def get_metrics(self):
        """
        Returns the evaluation metrics of the tested model.

        Returns:
            dict: Metrics or None if not available
        """
        if self.current_model is None:
            return None

        return self.current_model.metrics

    def get_model_feature_count(self):
        """
        Returns the number of input features the current model expects.

        Returns:
            int or None: Feature count, or None if no fitted model is available.
        """
        if self.current_model is None or self.current_model.model is None:
            return None

        return getattr(self.current_model.model, "n_features_in_", None)

    def predict(self, X):
        """
        Runs prediction with the current model on the given feature matrix.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            dict: {"success": bool, "message": str, "predictions": array or None}
        """
        if self.current_model is None or self.current_model.model is None:
            return {"success": False, "message": "No model loaded", "predictions": None}

        try:
            predictions = self.current_model.model.predict(X)
            return {"success": True, "message": "Prediction successful", "predictions": predictions}
        except Exception as e:
            return {"success": False, "message": f"Prediction failed: {str(e)}", "predictions": None}

    def get_confusion_matrix_figure(self):
        """
        Builds the confusion matrix visualization for the tested model.

        Returns:
            matplotlib.figure.Figure or None: The figure, or None if unavailable.
        """
        if self.current_model is None or self.current_model.metrics["conf_matrix"] is None:
            return None

        try:
            return self.current_model.get_confusion_matrix_figure()
        except Exception:
            return None

    def save_model(self, file_path):
        """
        Saves the trained model to a file.

        Args:
            file_path: Path where to save the model

        Returns:
            dict: {"success": bool, "message": str, "filename": str or None}
        """
        if self.current_model is None or self.current_model.model is None:
            return {"success": False, "message": "No trained model to save", "filename": None}

        try:
            with open(file_path, 'wb') as file:
                pickle.dump(self.current_model.model, file)

            filename = os.path.basename(file_path)
            return {"success": True, "message": f"Model saved as '{filename}'", "filename": filename}

        except Exception as e:
            return {"success": False, "message": f"Failed to save model: {str(e)}", "filename": None}

    def get_model_bytes(self):
        """
        Serializes the trained model to bytes, for in-memory download (e.g. Streamlit).

        Returns:
            bytes or None: The pickled model, or None if no trained model exists.
        """
        if self.current_model is None or self.current_model.model is None:
            return None

        return pickle.dumps(self.current_model.model)

    def load_model(self, file_path):
        """
        Loads a previously saved model.

        Args:
            file_path: Path to the saved model file, or a file-like object
                (e.g. a Streamlit UploadedFile) exposing a `.name` attribute

        Returns:
            dict: {"success": bool, "message": str, "model_info": dict or None}
        """
        is_file_like = hasattr(file_path, 'read')

        if not is_file_like and not os.path.exists(file_path):
            return {"success": False, "message": "File does not exist", "model_info": None}

        try:
            if is_file_like:
                loaded_model = pickle.load(file_path)
                filename = getattr(file_path, 'name', 'model.pkl')
            else:
                with open(file_path, 'rb') as file:
                    loaded_model = pickle.load(file)
                filename = os.path.basename(file_path)

            # Create a new model instance
            self.current_model = self.model_registry["linear_reg"]()  # Default type
            self.current_model.model = loaded_model
            
            # Extract model information
            model_info = {
                "class_name": loaded_model.__class__.__name__,
                "parameters": {}
            }
            
            # Extract specific parameters based on model type
            if hasattr(loaded_model, 'n_neighbors'):
                model_info["parameters"]["n_neighbors"] = loaded_model.n_neighbors
                
            if hasattr(loaded_model, 'max_depth'):
                model_info["parameters"]["max_depth"] = loaded_model.max_depth
                
            if hasattr(loaded_model, 'kernel'):
                model_info["parameters"]["kernel"] = loaded_model.kernel
                
            return {
                "success": True,
                "message": f"Model '{filename}' loaded successfully!",
                "model_info": model_info
            }
            
        except Exception as e:
            return {"success": False, "message": f"Failed to load model: {str(e)}", "model_info": None}
