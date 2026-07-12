import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input

def plot_confusion_matrix(y_test, y_pred, model_name):
    try:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        output_dir = Path(__file__).resolve().parents[1] / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{model_name}_confusion_matrix.png")
        plt.show()
        print(f"{model_name} confusion matrix saved!")
    except Exception as e:
        print(f"Error: {e}")

def plot_model_comparison(lr_acc, dt_acc, nn_acc):
    try:
        models = ['Logistic Regression', 'Decision Tree', 'Neural Network']
        accuracies = [lr_acc, dt_acc, nn_acc]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=models, y=accuracies)
        plt.title('Model Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center')
        plt.savefig('reports/model_comparison.png')
        plt.show()
        print("Model comparison chart saved!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[1]
    data_path = base_path / "data" / "cleaned_loan_data.csv"

    df = pd.read_csv(data_path)
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

    nn_model = keras.Sequential([
        Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)
    nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)[1]

    plot_confusion_matrix(y_test, lr_model.predict(X_test), "Logistic_Regression")
    plot_confusion_matrix(y_test, dt_model.predict(X_test), "Decision_Tree")
    plot_model_comparison(lr_acc, dt_acc, nn_acc)