# Diabetes Prediction for Females using Support Vector Machine (SVM)

## Overview
This project involves building a Support Vector Machine (SVM) model to predict diabetes in females. The dataset used contains various health metrics, and the goal is to determine whether an individual has diabetes based on these metrics.

## Dataset
The dataset used for this project is the "PIMA Indian Diabetes" dataset, which contains 768 instances and 9 attributes. The attributes include health metrics such as glucose levels, blood pressure, insulin levels, BMI, age, and more. The last column is the target variable indicating whether the individual has diabetes (1) or not (0).

## Prerequisites
To run this project, you need to have the following software installed:
- Python 3.x
- Jupyter Notebook (optional, but recommended for an interactive environment)
- Required Python libraries: `numpy`, `pandas`, `scikit-learn`

You can install the required Python libraries using the following command:
```bash
pip install numpy pandas scikit-learn
```

## Project Structure
The project directory contains the following files:
- `diabetes_prediction_svm.ipynb`: Jupyter Notebook containing the entire workflow for the SVM model.
- `diabetes.csv`: The dataset file containing the diabetes data.
- `README.md`: This readme file providing an overview of the project.

## Steps to Run the Project

1. **Clone the Repository**: Clone the project repository to your local machine.
    ```bash
    git clone https://github.com/Ravij-p/DiabetesPredictionSVM.git
    cd DiabetesPredictionSVM
    ```

2. **Import Dependencies**: Import the necessary libraries including `numpy`, `pandas`, `svm` from `sklearn`, `train_test_split`, `accuracy_score`, and `StandardScaler`.

3. **Load the Dataset**: Load the dataset using pandas.
    ```python
    dataset = pd.read_csv('diabetes.csv')
    ```

4. **Explore the Dataset**: Check the shape and preview the first few rows of the dataset to understand its structure and contents.

5. **Data Preprocessing**:
    - Split the data into features (`X`) and target (`y`).
    - Standardize the data using `StandardScaler` to ensure all features contribute equally to the model.

6. **Split the Data**: Split the dataset into training and testing sets using `train_test_split` with a test size of 20% and stratified sampling to maintain the proportion of classes.

7. **Train the Model**:
    - Train an SVM model using a linear kernel.
    - Fit the model on the training data.

8. **Evaluate the Model**:
    - Make predictions on the training data and calculate the accuracy.
    - Make predictions on the testing data and calculate the accuracy.
    - Print the accuracy scores for both training and testing data to assess the model's performance.

9. **Prediction for New Data**:
    - Use the trained model to make predictions on new input data.
    - Standardize the input data before making predictions.
    - Print the prediction result indicating whether the individual is diabetic or not.

## Example Output
The model achieved an accuracy of approximately 78.66% on the training data and 77.27% on the testing data. A sample prediction indicated that the individual is not diabetic based on the provided input data.

## Conclusion
This project demonstrates how to build and evaluate a Support Vector Machine (SVM) model for predicting diabetes in females. The provided Jupyter Notebook walks through the entire process, from loading the data to training the model and evaluating its performance.

Feel free to explore and modify the code to improve the model or try different machine learning algorithms.
