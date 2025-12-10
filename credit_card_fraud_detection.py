#Credit Card Fraud Detection-Task 2
#Written by Arshath Abdulla A

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_path = r"C:\Users\ARSHATH ABDULLA A\OneDrive\Desktop\Internships\CODSOFT\CODSOFT\Task2_Credit_cArd\fraudTrain.csv"
test_path  = r"C:\Users\ARSHATH ABDULLA A\OneDrive\Desktop\Internships\CODSOFT\CODSOFT\Task2_Credit_cArd\fraudTest.csv"

def load_data():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    return train_df, test_df

def detect_target_column(df):
    for name in ["Class", "isFraud", "is_fraud", "fraud", "target"]:
        if name in df.columns:
            print("Target column found:", name)
            return name
    raise Exception("No target column found!")

def preprocess(df):
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]
    df = df.fillna(0)
    return df

def prepare_data(train_df, test_df, target_col):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_train, y_train, X_test, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

if __name__ == "__main__":
    train_df, test_df = load_data()
    target = detect_target_column(train_df)
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    X_train, y_train, X_test, y_test = prepare_data(train_df, test_df, target)
    X_train, X_test = scale_features(X_train, X_test)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
