#IMPORT LIB
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

#1.Load the dataset
df = pd.read_csv('diabetes.csv')

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)


#2.Handling missing values
imputer = SimpleImputer(strategy='mean')
df[cols_with_zero] = imputer.fit_transform(df[cols_with_zero])

#3.Check if class is imbalanced
X = df.drop('Outcome', axis=1)
Y = df['Outcome']


#4.Apply SMOTE for handling class imbalance
smote = SMOTE()
transform_feature, transform_label = smote.fit_resample(X, Y)

#5.Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(transform_feature, transform_label, test_size=0.2, random_state=2, stratify=transform_label)


#6. Feature Scaling
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

#7. Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaler, Y_train)

#8. Prediction
y_pred = model.predict(X_test_scaler)

#9. Evaluation
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

#10. Save the model
with open('app/model.pkl', 'wb') as file:
    pickle.dump((scaler, model), file)


print("Model saved successfully as 'diabetes_model.pkl'")

