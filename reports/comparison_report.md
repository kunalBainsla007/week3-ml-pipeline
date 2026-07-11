# Model Comparison Report - Loan Approval Prediction

## 1. Problem Statement
Predict whether a loan application should be approved or rejected based on applicant details.

## 2. Dataset
- Source: Kaggle Loan Prediction Dataset
- Total Rows: 614
- Features: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
- Target: Loan_Status (Y/N)

## 3. Data Preprocessing
- Missing values handled using median and mode
- Label Encoding for binary categories (Gender, Married, Education, Self_Employed)
- One-Hot Encoding for Property_Area
- Dependents '3+' replaced with 3

## 4. Models Trained
### Logistic Regression (Baseline)
- Simple linear model
- Best for binary classification problems
- Fast and interpretable

### Decision Tree
- Tree based model
- Captures non-linear patterns
- Prone to overfitting on small datasets

### Neural Network (TensorFlow)
- 3 hidden layers — 256, 128, 64 neurons
- Dropout layers to prevent overfitting
- 200 epochs, batch size 16

## 5. Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 0.79 |
| Decision Tree | 0.68 |
| Neural Network | 0.65 |

## 6. Best Model
**Logistic Regression** performed best with 79% accuracy because:
- Dataset is small (614 rows) — simple models work better
- Features have linear relationship with target
- Neural Networks need large data to outperform classical ML

## 7. Failure Cases & Improvements
### Failure Cases
- Logistic Regression struggles to predict rejections (Class 0)
- Dataset is imbalanced — more approvals than rejections
- Neural Network underperforms due to small dataset size

### Suggested Improvements
- Use SMOTE to handle class imbalance
- Add more features like credit score, existing loans
- Try Random Forest or XGBoost for better results
- Collect more data for better Neural Network performance

## 8. Conclusion
Logistic Regression is the best model for this dataset. For larger datasets with more features, Neural Networks would likely outperform classical ML models.