# ModelMaster

ModelMaster is a Python-based GUI application designed to help users select the best classification algorithm for their dataset. The application provides an intuitive interface for loading datasets, selecting features, training models, and evaluating their performance.

## Features

- **Dataset Loader**: Load your preprocessed dataset in CSV format.
- **Feature Ranking**: Uses Recursive Feature Elimination (RFE) to rank the best features in your dataset.
- **Algorithm Selection**: Choose from the following classification algorithms:
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Support Vector Machine (SVM)
  - Logistic Regression
- **Model Training and Testing**: Train and test your model with a user-defined train-test split ratio.
- **Performance Metrics**: View accuracy, precision, recall, F1 score, and confusion matrix.
- **Model Saving and Loading**: Save trained models using the `pickle` library and reload them for evaluation.

## How It Works

1. **Load Dataset**: Start by loading your preprocessed dataset.
2. **Feature Ranking**: The app uses RFE to rank the best features in your dataset.
3. **Algorithm Selection**: Choose a classification algorithm to evaluate:
   - For KNN, specify the number of neighbors.
   - For Decision Tree, specify the maximum depth.
4. **Train and Test**: Specify the train-test split ratio and train your model.
5. **View Results**: The app displays performance metrics such as accuracy, precision, recall, F1 score, and confusion matrix.
6. **Save and Load Models**: Save your trained model for future use or load an existing model to view its details.

## Installation

```bash
git clone https://github.com/Lordiod/ModelMaster.git
cd ModelMaster
pip install -r requirements.txt
python model_master_gui.py
```

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- tkinter
- matplotlib

## Future Improvements

- Enhanced model loader with additional details about the saved model.
- Support for additional algorithms and preprocessing steps.
- Improved error handling and user feedback.

## Why Use ModelMaster?

This app simplifies the process of selecting and evaluating classification algorithms, helping you choose the best one for your dataset. Whether you're a beginner or an experienced data scientist, ModelMaster provides a user-friendly interface to streamline your workflow.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.