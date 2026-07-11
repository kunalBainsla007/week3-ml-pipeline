# Week 3 - ML Pipeline: Loan Approval Prediction

## Problem Statement
Predict whether a loan application should be approved or rejected based on applicant details.

## Models
- Logistic Regression (Baseline)
- Decision Tree
- Neural Network (TensorFlow)

## Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 0.79 |
| Decision Tree | 0.68 |
| Neural Network | 0.65 |

## How to Run

### Setup
```bash
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
```

### Run Pipeline
```bash
python src/pipeline.py
```

### Train Models
```bash
python src/train.py
```

### Evaluate Models
```bash
python src/evaluate.py
```

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow