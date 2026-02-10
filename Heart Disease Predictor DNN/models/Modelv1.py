import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

model = load_model('models/heart_disease_model.keras')
scaler = joblib.load('models/scaler.pkl')

df = pd.read_csv('data/Heart_Disease_Prediction.csv')

from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=43)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.15, random_state=43)

X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

def get_acc(X_scaled, y_true):
    probs = model.predict(X_scaled, verbose=0)
    preds = (probs > 0.5).astype("int32")
    return accuracy_score(y_true, preds)

print(f"Training Accuracy:   {get_acc(X_train_s, y_train):.4f}")
print(f"Validation Accuracy: {get_acc(X_val_s, y_val):.4f}")
print(f"Testing Accuracy:    {get_acc(X_test_s, y_test):.4f}")
