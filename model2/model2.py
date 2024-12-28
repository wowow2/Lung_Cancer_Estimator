'''
title: Lung Cancer Estimator
author: Abbas Rizvi
date: December 27th, 2024
description: This script builds and evaluates a lung cancer risk prediction model using XGBoost,
             with SHAP analysis for model interpretability. It includes data preprocessing,
             model training, evaluation metrics, and various visualization techniques.
'''

### --- Imports --- ###
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from xgboost import XGBClassifier
import joblib
import shap

# Reset matplotlib style to ensure consistent visualization
plt.style.use('default')

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

### --- Data Loading and Preprocessing --- ###
# Load the dataset
df = pd.read_excel('cancer patient data sets.xlsx')

# Convert categorical variables to numerical using Label Encoding
# This is necessary for machine learning models that require numerical inputs
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Level'] = label_encoder.fit_transform(df['Level'])  # Target variable encoding

# Separate features (X) and target variable (y)
# Remove 'Level' (target) and 'Patient Id' (non-predictive) from features
X = df.drop(['Level', 'Patient Id'], axis=1)
y = df['Level']

# Standardize features to have zero mean and unit variance
# This ensures all features are on the same scale and equally weighted
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Convert to DataFrame to preserve column names for interpretability
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training (80%) and testing (20%) sets
# random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

### --- Model Training --- ###
# Initialize XGBoost classifier
# XGBoost is chosen for its high performance and built-in feature importance capabilities
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
# Train the model on our preprocessed data
model.fit(X_train, y_train)

# Save trained model and scaler for future use
# This allows deployment without retraining
joblib.dump(model, 'lung_cancer_xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

### --- Model Evaluation --- ###
# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calculate AUC for multi-class classification
# 'ovr' means one-vs-rest approach for multi-class problems
auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
print(f'AUC (C-statistic): {auc:.2f}')

# Display detailed classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

### --- Visualization Functions --- ###
def plot_confusion_matrix():
    """
    Creates and saves a confusion matrix heatmap showing model's classification performance.
    The matrix shows true vs predicted values for each class.
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted Level')
    plt.ylabel('Actual Level')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_pca():
    """
    Performs PCA dimensionality reduction and creates a scatter plot showing
    the first two principal components. This helps visualize how well the
    classes can be separated in 2D space.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Level'] = y

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Level', palette='viridis', alpha=0.8)
    plt.title("PCA: Lung Cancer Risk Levels")
    plt.tight_layout()
    plt.savefig('pca_plot.png')
    plt.close()

def plot_shap_summary():
    """
    Creates and saves SHAP analysis plots:
    1. Summary plot: Shows the impact of each feature on model output
    2. Feature importance plot: Shows global feature importance

    SHAP values explain each prediction by attributing portions of the prediction
    to each input feature. This helps understand which features drive the model's decisions.
    """
    # Initialize SHAP explainer for tree models (XGBoost)
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for test set
    # These values show how each feature contributes to each prediction
    shap_values = explainer.shap_values(X_test)

    # Create and save summary plot
    # This shows how features impact the model output
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()

    # Create and save feature importance bar plot
    # This shows the average magnitude of feature impacts
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X_test.columns, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig('shap_importance.png')
    plt.close()

### --- Generate and Save All Plots --- ###
print("\nGenerating plots...")
plot_confusion_matrix()
print("Confusion matrix plot saved as 'confusion_matrix.png'")

plot_pca()
print("PCA plot saved as 'pca_plot.png'")

plot_shap_summary()
print("SHAP plots saved as 'shap_summary.png' and 'shap_importance.png'")

print("\nAll plots have been generated and saved to the current directory.")