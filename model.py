'''
title: Outcome Estimator for Lung Cancer
author: Abbas Rizvi
date: December 12th, 2024
'''

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


### --- Load Data --- ###

df = pd.read_csv('survey_lung_cancer.csv')

### --- Preprocess data --- ###

# first convert LUNG_CANCER and GENDER into 0s and 1s
df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({'YES': 1, 'NO': 0})
df['GENDER'] = df['GENDER'].replace({'M': 1, 'F': 0})

# age can be left as is, now we convert the other columns which are 1 (no) and 2 (yes) into 0s and 1s
binary_columns = df.columns.drop(['LUNG_CANCER', 'GENDER', 'AGE'])  # Exclude columns we have processed
for col in binary_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# drop not relevant columns
df = df.drop(columns=['GENDER','AGE', 'SMOKING', 'SHORTNESS OF BREATH'])

# split our features (X) and target (Y)
X = df.drop('LUNG_CANCER', axis=1)  # Features: Drop the target column
y = df['LUNG_CANCER']  # Target: Predict presence of lung cancer

# imbalanced sampling handling
adasyn = ADASYN(random_state=42)
X, y = adasyn.fit_resample(X, y)

# split data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### --- Model Training --- ###

# initialize model
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='auc')

# train model on training data
model.fit(X_train, y_train)

# evalulate on test data
y_pred = model.predict(X_test) # predict the test set
accuracy = accuracy_score(y_test, y_pred) # calculate accuracy
print(f'Accuracy: {accuracy:.2f}')

# Calculate AUC (C-statistic)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f'AUC (C-statistic): {auc:.2f}')

### --- Outputs --- ###

print("\nClassification Report:")
print(classification_report(y_test, y_pred)) # classification report

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred)) # confusion matrix

### --- Save Model --- ###
joblib.dump(model, 'lung_cancer_model.pkl')

### --- Predictions --- ###
def predict_lung_cancer(input_data):
    """
    Predict whether the user has lung cancer based on input features.
    :param input_data: List or array-like structure containing user inputs for all features
    :return: Prediction result (0 = No Lung Cancer, 1 = Lung Cancer)
    """
    # Load the trained model
    trained_model = joblib.load('lung_cancer_model.pkl')

    # Ensure input is a 2D array (required for prediction)
    input_data_reshaped = pd.DataFrame([input_data], columns=X.columns)

    # Make a prediction
    prediction = trained_model.predict(input_data_reshaped)
    return "Lung Cancer" if prediction[0] == 1 else "No Lung Cancer"


def get_user_input():
    """
    Ask the user for input for each feature and return it in the format needed for prediction.
    User must enter 0 or 1 directly.
    :return: A list of user inputs corresponding to the features.
    """
    print("Please answer the following questions with '0' (No) or '1' (Yes).")

    # Dictionary for each question and its corresponding index in the input list
    features = [
        "Do you have yellow fingers?",
        "Do you suffer from anxiety?",
        "Do you experience peer pressure?",
        "Do you have any chronic diseases?",
        "Do you experience fatigue?",
        "Do you have allergies?",
        "Do you experience wheezing?",
        "Do you consume alcohol?",
        "Do you have a persistent cough?",
        "Do you have swallowing difficulty?",
        "Do you experience chest pain?"
    ]

    # List to store the user inputs (1 = Yes, 0 = No)
    user_input = []

    for feature in features:
        while True:
            try:
                user_response = int(input(f"{feature} (0 = No, 1 = Yes): ").strip())
                if user_response == 0 or user_response == 1:
                    user_input.append(user_response)
                    break
                else:
                    print("Invalid input. Please enter '0' for No or '1' for Yes.")
            except ValueError:
                print("Invalid input. Please enter '0' or '1'.")

    return user_input


if __name__ == "__main__":
    user_input = get_user_input()  # Get the user input
    prediction = predict_lung_cancer(user_input)

    print("\nPrediction Result:")
    print(prediction)


















