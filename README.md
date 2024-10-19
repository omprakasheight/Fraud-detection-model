# Fraud Detection Model

This project is focused on building a machine learning model to detect fraudulent transactions. The project uses supervised learning techniques and includes steps such as data preprocessing, model training, and model evaluation.

## Project Overview

The main goal of this project is to predict fraudulent transactions based on historical data. The dataset used contains information about various transactions, including their amounts, origins, and destinations, and whether or not they were marked as fraud.

Key steps in the project:
1. **Data Collection**: The dataset is provided as a CSV file.
2. **Data Preprocessing**: Missing values are handled, features are scaled, and class imbalance is addressed using techniques like SMOTE.
3. **Model Training**: Machine learning models like Random Forest, Gradient Boosting, and XGBoost are used for training.
4. **Model Evaluation**: Performance metrics such as accuracy, precision, recall, F1-score, and AUC-ROC are calculated.

## Folder Structure

```
fraud_detection_project/
├── data/
│   └── fraud_data.csv              # Dataset used for training
├── src/
│   └── fraud_detection.py          # Main Python script for the project
├── notebooks/
│   └── fraud_detection.ipynb       # Jupyter notebook with code and analysis
└── README.md                       # Project documentation
```

## Setup Instructions

### Prerequisites

Before running the project, ensure you have the following tools installed:
- Python 3.x
- Jupyter Notebook (optional, for notebook-based development)
- Required Python packages

### Installing Required Libraries

You can install all the necessary Python packages using the following command:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the libraries manually:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn
```

### Running the Project

1. **Option 1: Running the Python Script**

   - Navigate to the project directory in the terminal:
     ```bash
     cd /path/to/fraud_detection_project
     ```
   - Run the `fraud_detection.py` script:
     ```bash
     python3 src/fraud_detection.py
     ```

   This script will load the dataset, preprocess the data, train the model, and output performance metrics.

2. **Option 2: Running the Jupyter Notebook**

   - Start Jupyter Notebook in your project directory:
     ```bash
     jupyter notebook
     ```
   - Open the `fraud_detection.ipynb` file and run the cells to execute the steps interactively.

## Key Files

- **fraud_detection.py**: Contains the code for loading the data, preprocessing it, training the model, and evaluating it.
- **fraud_data.csv**: The dataset used for training and evaluating the models.
- **fraud_detection.ipynb**: A notebook version of the project for easier visualization and step-by-step execution.

## Project Flow

1. **Data Loading**: The `fraud_data.csv` file is loaded and explored.
2. **Data Preprocessing**:
   - Handling missing values.
   - Scaling features.
   - Dealing with class imbalance using SMOTE.
3. **Model Training**:
   - Models like Random Forest, Gradient Boosting, and XGBoost are trained.
   - Hyperparameter tuning is performed on XGBoost.
4. **Model Evaluation**:
   - The models are evaluated based on metrics such as precision, recall, F1-score, and AUC-ROC.
   - Confusion matrices and ROC curves are visualized.

## Evaluation Metrics

- **Accuracy**: The overall performance of the model.
- **Precision**: How many of the predicted frauds were actually fraudulent.
- **Recall**: How many actual frauds were correctly detected.
- **F1-Score**: A balance between precision and recall.
- **AUC-ROC**: The area under the ROC curve, indicating the model's ability to distinguish between fraudulent and non-fraudulent transactions.

## Results

- The final models achieve a good balance between precision and recall, with XGBoost showing strong performance in detecting fraud.
- The feature importance analysis reveals which factors are most predictive of fraudulent behavior, such as transaction amounts and account history.

## Future Improvements

- Explore additional algorithms like neural networks for more complex fraud detection.
- Improve the handling of class imbalance using advanced techniques.
- Use real-time data processing for real-world fraud detection scenarios.
