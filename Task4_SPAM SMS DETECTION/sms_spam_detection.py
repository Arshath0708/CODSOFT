import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=5000
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))

def save_artifacts(model, vectorizer):
    joblib.dump(model, "spam_classifier_model.joblib")
    joblib.dump(vectorizer, "spam_vectorizer.joblib")

def main():
    dataset_path = r"C:\Users\ARSHATH ABDULLA A\OneDrive\Desktop\Internships\CODSOFT\CODSOFT\Task4 Spam\spam.csv"
    df = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = split_data(df)
    model, vectorizer = train_model(X_train, y_train)
    evaluate_model(model, vectorizer, X_test, y_test)
    save_artifacts(model, vectorizer)

if __name__ == "__main__":
    main()
