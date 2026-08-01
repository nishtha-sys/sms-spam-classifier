# 📩 SpamShield: Classical ML vs Transformers

Compares a classical ML pipeline (TF-IDF + Logistic Regression) against a fine-tuned DistilBERT transformer for SMS spam detection. 
A machine learning web app that classifies SMS messages as **Spam** or **Ham (Not Spam)** — with a live toggle to compare a classical ML pipeline against a fine-tuned transformer model.

🔗 **Live App:** [nishtha-sms-spam-classifier.streamlit.app](https://nishtha-sms-spam-classifier.streamlit.app)

---

## 📝 Project Description

This project detects spam SMS messages using two different approaches, side by side:

- **Fast model:** TF-IDF (unigrams + bigrams) + Logistic Regression
- **Accurate model:** Fine-tuned DistilBERT transformer, hosted on [Hugging Face Hub](https://huggingface.co/nishthasahani/sms-spam-distilbert)

Users can switch between the two in real time and compare predictions, confidence scores, and speed — a practical demonstration of the classical-ML-vs-transformer trade-off.

---

## ✨ Features

- Classifies any custom SMS/text message as **Spam** or **Ham**
- **Model toggle** — switch between Logistic Regression and DistilBERT instantly
- Color-coded results with a visual confidence gauge (green/orange/red based on risk level)
- "Try an example" buttons with randomized spam/ham sample messages
- Session history — tracks recent checks, tagged with which model was used
- Expandable "How this works" section explaining both models in plain language
- Sidebar with live project stats, tech stack, and links

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Classical ML:** scikit-learn — TF-IDF (unigrams + bigrams) + Logistic Regression
- **Deep Learning:** Hugging Face Transformers — fine-tuned DistilBERT
- **Web Framework:** Streamlit
- **Model Hosting:** Hugging Face Hub
- **Prototyping:** Google Colab, Gradio (initial demo)
- **Deployment:** Streamlit Community Cloud

---

## 📊 Model Performance

| Metric | Logistic Regression | DistilBERT |
|---|---|---|
| Accuracy | 98.3% | **99.0%** |
| Ham correctly identified | 958 | 962 |
| Ham misclassified as spam | 8 | **4** |
| Spam correctly identified | 138 | **143** |
| Spam missed | 11 | **6** |
| ROC-AUC | — | 0.996 |

**Iteration history:** the project started with unigram TF-IDF + Multinomial Naive Bayes (97.04% accuracy, 33 missed spam messages). Switching to bigram TF-IDF + Logistic Regression with class balancing cut missed spam to 11. Fine-tuning DistilBERT improved on both false positives and false negatives simultaneously — a rare case where a more complex model didn't require a trade-off.

**Known limitation:** the training data (UCI SMS Spam Collection) was compiled around 2011-2012 and reflects spam patterns from that era (premium-rate scams, ringtone subscriptions, etc.). Modern smishing patterns (phishing links, fake delivery notifications, OTP fraud) are underrepresented, so both models may underperform on contemporary spam styles not well reflected in this dataset.

---

## 📂 Dataset

[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

---

## 📁 Project Structure

```
sms-spam-classifier/
├── app.py                          # Streamlit app with dual-model toggle
├── spam_model.pkl                  # Pre-trained TF-IDF + Logistic Regression pipeline
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── SMS_Spam_Classifier.ipynb   # Full training workflow (source of truth)
├── README.md
└── .gitignore
```

> **Note:** The DistilBERT model (~268MB) is hosted on Hugging Face Hub rather than committed to this repo, and is loaded at runtime. The app only needs `app.py`, `spam_model.pkl`, and `requirements.txt` to run.

---

## 💻 Run Locally

1. Clone the repository
   ```bash
   git clone https://github.com/nishtha-sys/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at `http://localhost:8501`

---

## 🚀 Future Improvements

- Add a FastAPI backend + database to log predictions and separate the ML layer from the UI
- Expand training data with more contemporary spam/smishing examples
- Add batch message upload/testing