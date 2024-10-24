# House-Prices-Prediction

This repository contains a machine learning project focused on predicting house prices using the **Random Forest Regressor** algorithm. The project leverages a dataset containing various features related to houses, such as their size, location, and condition, to estimate their market prices.

## Introduction

Predicting house prices is a common task in the field of machine learning, with numerous applications in real estate and finance. This project aims to build an effective predictive model that can generalize well on unseen data. The Random Forest algorithm was chosen due to its ability to handle both linear and non-linear data, as well as its robustness in dealing with missing and categorical values.

By the end of this project, we expect the model to predict house prices with reasonable accuracy, while providing insights into which features influence the final price.

## Dataset Overview

The dataset contains detailed information about houses, with both a training set (for training the model) and a test set (for generating predictions). The training set includes the house prices as the target variable, which is used to train the model, while the test set is used to evaluate its performance.

## Preprocessing Steps

To prepare the data for modeling, several preprocessing steps were applied:

- **Handling Missing Values**: Missing numerical values were imputed using the median of the respective feature, while categorical missing values were filled with 'None' to signify the absence of specific information.
  
- **Label Encoding**: Categorical features such as `Neighborhood`, `HouseStyle`, and `SaleCondition` were encoded into numeric values using label encoding. This allows the Random Forest model to process categorical data effectively.
  
- **Feature Selection**: Features that were irrelevant or redundant, such as the `Id` column, were dropped from the dataset. This helped streamline the model and avoid overfitting.

- **Data Scaling**: Although Random Forest does not require feature scaling, we ensured numerical features were appropriately normalized where necessary to improve model efficiency.

## Algorithm Used

- **Random Forest Regressor**: A Random Forest model was implemented to predict house prices. This algorithm creates multiple decision trees during training and merges them to produce more accurate and stable predictions. Random Forest is particularly useful in dealing with complex datasets and reducing variance, thereby minimizing overfitting.

  - The model was trained on the training data using k-fold cross-validation to fine-tune hyperparameters and ensure generalization.
  - Features were ranked based on their importance, allowing us to understand which factors most influence house prices.

## Installation

To run this project locally, ensure you have Python installed along with the necessary libraries. You can install the required dependencies by running:

```bash
pip install pandas scikit-learn numpy matplotlib seaborn
```
## Usage

1. Download the **House Prices** dataset from Kaggle (or use local copies named `train.csv` and `test.csv`) and place them in the project directory.

2. Run the Python script `house_prices_model.py` to preprocess the data, train the model, and generate predictions:

   ```bash
   python code.py
   ```
The output file containing predictions for the test data will be saved as submission.csv.

You can also view model performance metrics such as Root Mean Squared Error (RMSE) and feature importance after running the script.

## Results
After training and validating the Random Forest Regressor model, the following performance metrics were achieved:

* Validation RMSE: 0.14973
* Cross-Validation Score: The model was cross-validated using a 5-fold split to reduce variance and improve generalization.
* Feature Importance: Features like OverallQual, GrLivArea, and TotalBsmtSF were identified as the most important factors in determining house prices.
## Conclusion
This project successfully demonstrated the use of the Random Forest algorithm for predicting house prices based on a variety of features. While the model achieved reasonable accuracy with an RMSE of 0.14973, there are still areas where improvements can be made, particularly through hyperparameter tuning and feature engineering. 
