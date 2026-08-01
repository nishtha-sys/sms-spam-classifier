# 📩 SpamShield: Classical ML vs Transformers

A machine learning web app that classifies SMS messages as **Spam** or **Ham (Not Spam)** — with a live toggle between a classical ML pipeline and a fine-tuned transformer, explainability, batch processing, and confidence-aware predictions.

🔗 **Live App:** [nishtha-sms-spam-classifier.streamlit.app](https://nishtha-sms-spam-classifier.streamlit.app)

---

## 📝 Project Description

This project detects spam SMS messages using two different approaches, side by side:

- **Fast model:** TF-IDF (unigrams + bigrams) + Logistic Regression
- **Accurate model:** Fine-tuned DistilBERT transformer, hosted on [Hugging Face Hub](https://huggingface.co/nishthasahani/sms-spam-distilbert)

Beyond classification, the app explains *why* a message was flagged, handles uncertain/borderline predictions honestly instead of forcing a binary answer, and supports checking many messages at once via file upload.

---

## ✨ Features

- Classifies any custom SMS/text message as **Spam** or **Ham**
- **Model toggle** — switch between Logistic Regression and DistilBERT instantly
- **Explainability** — shows the specific words that most influenced a spam prediction (Logistic Regression)
- **Confidence-aware predictions** — messages with 45-55% spam probability are flagged as "Uncertain — needs review" instead of forced into a binary label
- **Batch processing** — upload a CSV, TXT, or PDF file of messages, classify them all, and download results as CSV
- Color-coded results with a visual confidence gauge (green/orange/red based on risk level)
- "Try an example" buttons with randomized spam/ham sample messages
- Session history — tracks recent checks, tagged with which model was used
- Expandable "How this works" section explaining both models in plain language

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Classical ML:** scikit-learn — TF-IDF (unigrams + bigrams) + Logistic Regression
- **Deep Learning:** Hugging Face Transformers — fine-tuned DistilBERT
- **PDF parsing:** pypdf
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
| Inference speed | Instant | Slower (transformer) |

**Iteration history:** the project started with unigram TF-IDF + Multinomial Naive Bayes (97.04% accuracy, 33 missed spam messages). Switching to bigram TF-IDF + Logistic Regression with class balancing cut missed spam to 11. Fine-tuning DistilBERT improved on both false positives and false negatives simultaneously.

**Known limitation:** the training data (UCI SMS Spam Collection) was compiled around 2011-2012 and reflects spam patterns from that era (premium-rate scams, ringtone subscriptions, etc.). Modern smishing patterns (phishing links, fake delivery notifications, OTP fraud) are underrepresented, so both models may underperform on contemporary spam styles not well reflected in this dataset.

---

## 📂 Dataset

[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

---

## 📁 Project Structure

```
sms-spam-classifier/
├── app.py                          # Streamlit app: dual-model toggle, explainability, batch upload
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

- **FastAPI backend + database** — separate the ML layer from the UI, log predictions with timestamps
- **Dockerize** the application for consistent, portable deployment
- **CI/CD pipeline** (GitHub Actions) for automated testing and deployment
- **In-app evaluation dashboard** — precision, recall, F1, ROC curve, confusion matrix visualized live
- **Feedback loop** — let users flag misclassifications and use that to inform future retraining
- **Monitoring** — track latency, confidence distribution, and request volume over time
- Expand training data with more contemporary spam/smishing examples