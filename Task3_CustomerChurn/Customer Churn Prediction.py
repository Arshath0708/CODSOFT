import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\ARSHATH ABDULLA A\OneDrive\Desktop\Internships\CODSOFT\CODSOFT\Task2_Customer\churn_data.csv"

def load_data(path):
    df = pd.read_csv(path)
    print("Loaded data shape:", df.shape)
    return df

def basic_cleaning(df):
    to_drop = []
    for c in ["RowNumber", "CustomerId", "Surname"]:
        if c in df.columns:
            to_drop.append(c)
    if to_drop:
        df = df.drop(columns=to_drop)
        print("Dropped columns:", to_drop)
    df.columns = [c.strip() for c in df.columns]
    return df

def preprocess_features(df):
    target_col = "Exited"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found.")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c]).codes 
    X = X.fillna(0)
    return X, y
def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, scaler

def train_and_compare(X_train, y_train, X_test, y_test):
    logit = LogisticRegression(max_iter=1000, random_state=42)
    logit.fit(X_train, y_train)
    pred_logit = logit.predict(X_test)
    acc_logit = accuracy_score(y_test, pred_logit)

    print("\nLogistic Regression accuracy: {:.4f}".format(acc_logit))
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)
    print("RandomForest accuracy: {:.4f}".format(acc_rf))
    if acc_rf >= acc_logit:
        best_model = rf
        best_pred = pred_rf
        chosen = "RandomForest"
    else:
        best_model = logit
        best_pred = pred_logit
        chosen = "LogisticRegression"
    print(f"\nChosen model: {chosen}")
    print("\nClassification report (chosen model):\n")
    print(classification_report(y_test, best_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, best_pred))
    return best_model

def save_artifacts(model, scaler):
    joblib.dump(model, "churn_model.joblib")
    joblib.dump(scaler, "churn_scaler.joblib")
    print("\nSaved model -> churn_model.joblib")
    print("Saved scaler -> churn_scaler.joblib")

def main():
    df = load_data(DATA_PATH)
    df = basic_cleaning(df)
    X, y = preprocess_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    best = train_and_compare(X_train, y_train, X_test, y_test)
    save_artifacts(best, scaler)

if __name__ == "__main__":
    main()
