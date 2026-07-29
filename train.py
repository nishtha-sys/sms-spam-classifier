from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_FILE = PROJECT_DIR / "spam_model.pkl"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/228/"
    "sms+spam+collection.zip"
)


def download_dataset():
    """Download and extract the UCI SMS Spam Collection dataset."""
    if DATA_FILE.exists():
        return

    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "sms_spam_collection.zip"

    print("Downloading dataset...")
    urlretrieve(DATASET_URL, zip_path)

    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(DATA_DIR)

    zip_path.unlink()
    print("Dataset downloaded successfully.")


def main():
    download_dataset()

    data = pd.read_csv(
        DATA_FILE,
        sep="\t",
        names=["label", "text"],
        header=None
    )

    data["label_number"] = data["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label_number"],
        test_size=0.2,
        random_state=42,
        stratify=data["label_number"]
    )

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english")),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.2%}\n")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Ham", "Spam"]
    ))

    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
