import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

def clean_genre(raw_genre):
    if "drama" in raw_genre.lower():
        return "drama"
    elif "comedy" in raw_genre.lower():
        return "comedy"
    elif "action" in raw_genre.lower():
        return "action"
    elif "romance" in raw_genre.lower():
        return "romance"
    elif "documentary" in raw_genre.lower():
        return "documentary"
    elif "horror" in raw_genre.lower():
        return "horror"
    elif "thriller" in raw_genre.lower():
        return "thriller"
    elif "adult" in raw_genre.lower():
        return "adult"
    else:
        return "other"
def load_train_data(path):
    genres = []
    plots = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t")
            elif "|" in line:
                parts = line.split("|")
            else:
                parts = line.split(",")
            if len(parts) >= 2:
                genres.append(clean_genre(parts[0]))
                plots.append(parts[1])
    df = pd.DataFrame({"genre": genres, "plot": plots})
    print("Training data loaded:", df.shape)
    print(df.head())
    return df
def load_test_data(plot_path, label_path):
    with open(plot_path, "r", encoding="utf-8") as f:
        plots = [line.strip() for line in f if line.strip()]
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [clean_genre(line.strip()) for line in f if line.strip()]
    print("Test data loaded:", len(plots), "plots,", len(labels), "labels")
    return plots, labels

def train_and_evaluate(train_df, test_plots, test_labels):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train = vectorizer.fit_transform(train_df["plot"])
    y_train = train_df["genre"]

    model = MultinomialNB()
    model.fit(X_train, y_train)

    X_test = vectorizer.transform(test_plots)
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(test_labels, y_pred))
    print("\nClassification Report:\n", classification_report(test_labels, y_pred))

    joblib.dump((model, vectorizer), "movie_genre_model.joblib")
    print("Saved model to movie_genre_model.joblib")

if __name__ == "__main__":
    train_df = load_train_data("train_data.txt")
    test_plots, test_labels = load_test_data("test_data.txt", "test_data_solution.txt")
    train_and_evaluate(train_df, test_plots, test_labels)