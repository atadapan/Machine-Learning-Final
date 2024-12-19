import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('filtered_evt.csv')

# Selecting the relevant features
features = ['gfa', 'numfloors', 'year', 'natural_gas', 'electricity', 'bldgtype']
target_eui = 'eui'
threshold = 100  # Example threshold for energy efficiency

# Dropping rows with missing EUI values
data = data.dropna(subset=[target_eui])

# Creating binary target for energy efficiency classification
data['efficient'] = (data[target_eui] < threshold).astype(int)

# Splitting features and targets
X = data[features]
y_reg = data[target_eui]
y_class = data['efficient']

# Handling numerical and categorical columns
num_features = ['gfa', 'numfloors', 'year', 'natural_gas', 'electricity']
cat_features = ['bldgtype']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

# Preprocessing the features
X = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
    X, y_reg, y_class, test_size=0.2, random_state=42
)

# Linear Regression Model for EUI prediction
reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

# Logistic Regression Model for energy efficiency classification
class_model = LogisticRegression()
class_model.fit(X_train, y_class_train)
y_class_pred = class_model.predict(X_test)

# Metrics for Regression
reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
print(f"Mean Absolute Error for EUI Prediction: {reg_mae}")

# Metrics for Classification
class_accuracy = accuracy_score(y_class_test, y_class_pred)
print(f"Accuracy for Energy Efficiency Classification: {class_accuracy}")
print("\nClassification Report:")
print(classification_report(y_class_test, y_class_pred))

# Visualizing Actual vs Predicted EUI
plt.scatter(y_reg_test, y_reg_pred, alpha=0.5)
plt.title('Actual vs Predicted EUI')
plt.xlabel('Actual EUI')
plt.ylabel('Predicted EUI')
plt.show()

# Visualizing Classification Results
results_df = pd.DataFrame({
    'Actual_EUI': y_reg_test.reset_index(drop=True),
    'Predicted_EUI': y_reg_pred,
    'Actual_Efficiency': y_class_test.reset_index(drop=True),
    'Predicted_Efficiency': y_class_pred
})

# Displaying the first few rows of predictions
print(results_df.head())
