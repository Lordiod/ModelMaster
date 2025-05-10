# ModelMaster

ModelMaster is a Python-based GUI application designed to help users select the best classification algorithm for their dataset. The application provides an intuitive interface for loading datasets, selecting features, training models, and evaluating their performance.

![Model Master](qr-code.png)

## Features

- **Dataset Loader**: Load your preprocessed dataset in CSV format.
- **Feature Ranking**: Uses Recursive Feature Elimination (RFE) to rank the best features in your dataset.
- **Algorithm Selection**: Choose from the following classification algorithms:
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Linear Regression
- **Model Training and Testing**: Train and test your model with a user-defined train-test split ratio.
- **Performance Metrics**: View accuracy, precision, recall, F1 score, and confusion matrix.
- **Model Saving and Loading**: Save trained models using the `pickle` library and reload them for evaluation.

## How It Works

1. **Load Dataset**: Start by loading your preprocessed dataset.
2. **Feature Ranking**: The app uses RFE to rank the best features in your dataset.
3. **Algorithm Selection**: Choose a classification algorithm to evaluate:
   - For KNN, specify the number of neighbors.
   - For Decision Tree, specify the maximum depth.
   - For SVM, select a kernel type (linear, rbf, poly, or sigmoid).
4. **Train and Test**: Specify the train-test split ratio and train your model.
5. **View Results**: The app displays performance metrics such as accuracy, precision, recall, F1 score, and confusion matrix.
6. **Save and Load Models**: Save your trained model for future use or load an existing model to view its details.

## Project Structure

The project follows the Model-View-Controller (MVC) architecture:

```
ModelMaster/
│
├── main.py                    # Main entry point for the application
├── README.md                  # Project documentation
├── requirements.txt           # Project dependencies
│
├── src/                       # Source code directory
│   ├── model/                 # Model components
│   │   ├── __init__.py
│   │   └── ml_models.py       # Machine learning model implementations
│   │
│   ├── view/                  # View components
│   │   ├── __init__.py
│   │   └── gui.py             # GUI implementation
│   │
│   └── controller/            # Controller components
│       ├── __init__.py
│       └── model_controller.py # Controller logic
│
├── tests/                     # Test directory
│   ├── __init__.py
│   ├── test_models.py         # Tests for model components
│   └── test_controller.py     # Tests for controller components
│
├── Datasets/                  # Sample datasets for testing
└── .gitignore                 # Git ignore file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Lordiod/ModelMaster.git
cd ModelMaster

# Create a virtual environment (optional, but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- customtkinter (for modern GUI elements)
- matplotlib

## Development

### Testing

To run tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=src
```

## Future Improvements

- Enhanced model loader with additional details about the saved model
- Support for additional algorithms and preprocessing steps
- Improved error handling and user feedback
- Data visualization tools for exploratory data analysis
- Hyperparameter tuning automation
- Export functionality for model comparisons

## Why Use ModelMaster?

This app simplifies the process of selecting and evaluating classification algorithms, helping you choose the best one for your dataset. Whether you're a beginner or an experienced data scientist, ModelMaster provides a user-friendly interface to streamline your workflow.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.