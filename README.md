# mn-project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load Dataset
# You can download a dataset like US Accidents (https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
data = pd.read_csv("US_Accidents_Dec21_updated.csv")

# 2. Preprocessing
# Keeping only essential columns
columns = ['Severity', 'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition']
data = data[columns].dropna()

# Encode categorical data
data['Weather_Condition'] = data['Weather_Condition'].astype('category').cat.codes

# 3. Features and Target
X = data.drop('Severity', axis=1)
y = data['Severity']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Feature Importance
plt.figure(figsize=(10, 6))
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind='barh')
plt.title("Feature Importance in Accident Severity Prediction")
plt.tight_layout()
plt.show()
