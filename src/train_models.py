"""
AI in Cyber Security Lab - Unit 1 Mini Project
Student: Bhaumik Senwal | Roll No: 2301730328
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

df = pd.read_csv("data/cyber_login_events.csv")
X, y = df.drop(columns=["suspicious"]), df["suspicious"]
categorical = ["country", "device"]
numeric = [c for c in X.columns if c not in categorical]
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# Balance only the training set; keep the test set untouched.
train_df = X_train.copy()
train_df["suspicious"] = y_train.values
majority = train_df[train_df.suspicious == 0]
minority = train_df[train_df.suspicious == 1]
minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
balanced_train = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)
X_train = balanced_train.drop(columns="suspicious")
y_train = balanced_train["suspicious"]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(32,16), activation="relu",
        solver="adam", max_iter=300, random_state=42, early_stopping=True)
}
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(name)
    print("Accuracy:", round(accuracy_score(y_test,pred),4))
    print("Precision:", round(precision_score(y_test,pred,zero_division=0),4))
    print("Recall:", round(recall_score(y_test,pred,zero_division=0),4))
    print("F1:", round(f1_score(y_test,pred,zero_division=0),4))
