# AI Cybersecurity Unit-1 Mini Project

**Student:** Bhaumik Senwal  
**Roll No:** 2301730328  
**Course:** AI in Cyber Security Lab (ENSP355)

## Project
End-to-End Mini Project: Building an AI-Driven Cyber Threat Awareness and Detection Prototype.

## Structure
- `data/cyber_login_events.csv` - reproducible synthetic login-event dataset
- `src/train_models.py` - preprocessing, Logistic Regression and MLP implementation
- `outputs/` - EDA plots, confusion matrix, threshold analysis and results
- `requirements.txt` - Python dependencies

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train_models.py
```

## Methodology
Missing numeric values are median-imputed, categorical variables are most-frequent imputed and one-hot encoded, and numerical variables are standardized. Logistic Regression is used as the baseline supervised ML model and an MLPClassifier is used as the neural-network prototype.

## Dataset note
The dataset is synthetic and generated for this educational prototype. It represents login/security-event attributes such as failed attempts, access hour, country, device, VPN use, new-device status and IP reputation. No real personal data is included.

## Results
Logistic Regression: Accuracy=0.877, Precision=0.833, Recall=0.033, F1=0.064.

MLP Neural Network: Accuracy=0.875, Precision=1.000, Recall=0.007, F1=0.013.

## Ethics
Security classifiers can create false positives, false negatives, privacy risks and biased outcomes. Thresholds should be selected with operational context, human review and monitoring rather than treated as automatic proof of malicious activity.

## Academic integrity
This project uses a clearly identified synthetic dataset. If the submitted work is extended with an external dataset, cite its source in the report and repository.
