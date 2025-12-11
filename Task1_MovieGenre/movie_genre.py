import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib


def normalize_genre(text):
    text = text.lower()
    if "drama" in text:
        return "drama"
    if "comedy" in text:
        return "comedy"
    if "action" in text:
        return "action"
    if "romance" in text:
        return "romance"
    if "documentary" in text:
        return "documentary"
    if "horror" in text:
        return "horror"
    if "thriller" in text:
        return "thriller"
    if "adult" in text:
        return "adult"
    return "other"


def fetch_training_data(path):
    g_list = []
    p_list = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if "\t" in line:
                segment = line.split("\t")
            elif "|" in line:
                segment = line.split("|")
            else:
                segment = line.split(",")

            if len(segment) >= 2:
                g_list.append(normalize_genre(segment[0]))
                p_list.append(segment[1])

    df = pd.DataFrame({"genre": g_list, "plot": p_list})
    print("Loaded training data:", df.shape)
    print(df.head())
    return df


def fetch_test_data(plot_file, label_file):
    with open(plot_file, "r", encoding="utf-8") as f:
        test_plots = [line.strip() for line in f if line.strip()]

    with open(label_file, "r", encoding="utf-8") as f:
        test_labels = [normalize_genre(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(test_plots)} plots and {len(test_labels)} labels")
    return test_plots, test_labels


def train_model(train_data, eval_plots, eval_labels):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train = vectorizer.fit_transform(train_data["plot"])
    y_train = train_data["genre"]

    model = MultinomialNB()
    model.fit(X_train, y_train)

    X_eval = vectorizer.transform(eval_plots)
    predictions = model.predict(X_eval)

    print("\nAccuracy:", accuracy_score(eval_labels, predictions))
    print("\nClassification Summary:\n", classification_report(eval_labels, predictions))

    joblib.dump((model, vectorizer), "movie_genre_model.joblib")
    print("Model stored as movie_genre_model.joblib")


if __name__ == "__main__":
    train_df = fetch_training_data("train_data.txt")
    test_plots, test_labels = fetch_test_data("test_data.txt", "test_data_solution.txt")
    train_model(train_df, test_plots, test_labels)