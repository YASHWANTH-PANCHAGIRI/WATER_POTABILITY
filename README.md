# WATER_POTABILITY
USING MACHINE LEARNING ALGORITHMS


# Water Potability Prediction - GitHub README

## Project Overview
This project focuses on predicting the potability of water using various machine learning algorithms. The dataset contains multiple features that indicate water quality, and the goal is to build classification models that can accurately predict whether water is potable (drinkable) or not.

The code includes the use of logistic regression, K-Nearest Neighbors (KNN), Decision Trees, Support Vector Machines (SVM), Random Forests, Gradient Boosting, and XGBoost to build and evaluate different models. Additionally, data preprocessing techniques, such as handling missing values, normalization, and oversampling using SMOTE, are implemented.

## Dataset
The dataset used in this project is the **Water Potability Dataset**, which contains features related to water quality, such as pH, hardness, and various chemical components. The target variable is `Potability`, indicating whether the water is safe to drink.

## Key Features
- **Data Preprocessing**:
  - Handling missing values by filling with mean values.
  - Feature scaling using `MinMaxScaler`.
  - Addressing data imbalance with SMOTE.

- **Machine Learning Models**:
  - **Logistic Regression**: A baseline classification model.
  - **K-Nearest Neighbors (KNN)**: A distance-based algorithm to classify water potability.
  - **Decision Tree**: A tree-based model for classification.
  - **Support Vector Machine (SVM)**: Implemented with different kernels (linear, RBF, and polynomial).
  - **Random Forest**: An ensemble method for improving classification accuracy.
  - **Gradient Boosting**: A boosting technique to enhance model performance.
  - **XGBoost**: An optimized gradient boosting model.

- **Model Evaluation**:
  - Confusion matrix, classification report, and accuracy score are used to evaluate model performance.
  - Hyperparameter tuning for models like KNN (varying `n_neighbors`).
  
- **Visualization**:
  - Heatmaps for feature correlation.
  - Accuracy plots for KNN with varying K values.
  - Confusion matrix visualization.

## Code Structure

1. **Data Loading & Preprocessing**:
   - Load the dataset (`water_potability.csv`).
   - Handle missing values by filling with the mean.
   - Normalize feature values using `MinMaxScaler`.

2. **Logistic Regression**:
   - Train-test split.
   - Train the logistic regression model.
   - Evaluate performance using accuracy score and classification report.

3. **K-Nearest Neighbors (KNN)**:
   - Train a KNN model with different K values.
   - Plot accuracy vs K value.

4. **Decision Tree**:
   - Train a decision tree model.
   - Evaluate performance with accuracy, precision, recall, and F1 score.

5. **Support Vector Machine (SVM)**:
   - Train SVM with different kernels (linear, RBF, polynomial).
   - Evaluate model performance.

6. **Random Forest**:
   - Train a Random Forest classifier.
   - Evaluate performance using accuracy and other metrics.

7. **Gradient Boosting & XGBoost**:
   - Train Gradient Boosting and XGBoost classifiers.
   - Evaluate and compare accuracy for both models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YASHWANTH-PANCHAGIRI/water-potability-prediction.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd water-potability-prediction
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the dataset (`water_potability.csv`) in the project directory.

## Usage

1. Run the script to train and evaluate different models:
   ```bash
   python water_potability_prediction.py
   ```

2. Adjust hyperparameters or models in the script to experiment with different algorithms and configurations.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- XGBoost
- imbalanced-learn

## Results
The final models provide different levels of accuracy in predicting water potability. Here are the approximate accuracies of the models:
- Logistic Regression: ~65%
- KNN: ~65-70% (varies with `K` value)
- Decision Tree: ~70%
- SVM (RBF kernel): ~65%
- Random Forest: ~72%
- Gradient Boosting: ~72%
- XGBoost: ~74%

## Contributing
Feel free to contribute to this project by submitting a pull request. Please ensure that your changes are well-documented and tested.

## License
This project is licensed under the MIT License.

