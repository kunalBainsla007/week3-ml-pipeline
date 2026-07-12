import pandas as pd
from pathlib import Path

def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Data loaded — {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None

def clean_data(df):
    try:
        df = df.copy()
        df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
        df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
        df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
        df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
        df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
        df = df.drop(columns=['Loan_ID'], errors='ignore')
        return df
    except Exception as e:
        print(f"Error in cleaning: {e}")
        return None

def feature_engineering(df):
    try:
        df = df.copy()
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
        df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
        df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
        df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
        df = pd.get_dummies(df, columns=['Property_Area'])
        return df
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return None

def save_cleaned_data(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def run_pipeline(path):
    df = load_data(path)
    if df is None:
        return None
    df = clean_data(df)
    if df is None:
        return None
    df = feature_engineering(df)
    return df

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[1]
    input_path = base_path / "data" / "train.csv"
    output_path = base_path / "data" / "cleaned_loan_data.csv"
    
    df = run_pipeline(input_path)
    if df is not None:
        save_cleaned_data(df, output_path)
        print("Pipeline completed!")
        print(df.shape)
        print(df.head())