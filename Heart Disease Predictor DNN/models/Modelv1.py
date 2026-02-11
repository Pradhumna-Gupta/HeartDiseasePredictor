import os
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, 'heart_model.keras')
scaler_path = os.path.join(base_path, 'scaler.pkl')
data_path = os.path.join(base_path, '..', 'data', 'Heart_Disease_Prediction.csv')

model = load_model(model_path)
scaler = joblib.load(scaler_path)

df = pd.read_csv(data_path)

if 'Heart Disease' in df.columns:
    df['target'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    X = df.drop(['Heart Disease', 'target'], axis=1)
else:
    X = df.drop('target', axis=1)

Y = df['target']

X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size=0.1 , random_state = 42)
X_train_ , X_train_val , Y_train_ , Y_train_val = train_test_split(X_train , Y_train,test_size=0.1,random_state=42)
X_train_ = scaler.fit_transform(X_train_)
X_train_val = scaler.transform(X_train_val)
X_test = scaler.transform(X_test)

def get_acc(X_scaled, y_true):
    probs = model.predict(X_scaled, verbose=0)
    preds = (probs > 0.5).astype("int32")
    return accuracy_score(y_true, preds)

print("\n" + "="*30)
print(f"TRAINING ACCURACY:   {get_acc(X_train_, Y_train_)*100:.2f}%")
print(f"VALIDATION ACCURACY: {get_acc(X_train_val, Y_train_val)*100:.2f}%")
print(f"TESTING ACCURACY:    {get_acc(X_test, Y_test)*100:.2f}%")
print("="*30)

import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
cm = confusion_matrix(Y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Sick'],
            yticklabels=['Healthy', 'Sick'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Heart Disease Confusion Matrix')
plt.show()
