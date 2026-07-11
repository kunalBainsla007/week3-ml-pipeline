import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input

def load_cleaned_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Data loaded — {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None

def prepare_data(df):
    try:
        X = df.drop(columns=['Loan_Status'])
        y = df['Loan_Status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error in preparing data: {e}")
        return None

def train_logistic_regression(X_train, y_train):
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        print("Logistic Regression trained!")
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None

def train_decision_tree(X_train, y_train):
    try:
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("Decision Tree trained!")
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None

def train_neural_network(X_train, y_train, X_test, y_test):
    try:
        model = keras.Sequential([
            Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)
        result = model.evaluate(X_test, y_test, verbose=0)
        accuracy = result[1]
        print(f"\nNeural Network Accuracy: {accuracy:.2f}")
        return model, accuracy
    except Exception as e:
        print(f"Error in Neural Network: {e}")
        return None, None

def evaluate_model(model, X_test, y_test, model_name):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{model_name} Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))
        return accuracy
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[1]
    data_path = base_path / "data" / "cleaned_loan_data.csv"
    
    df = load_cleaned_data(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    lr_model = train_logistic_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    
    lr_accuracy = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    dt_accuracy = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    
    nn_model, nn_accuracy = train_neural_network(X_train, y_train, X_test, y_test)