# 📩 SMS Spam Classifier

A machine learning web app that classifies SMS messages as **Spam** or **Ham (Not Spam)**, built with scikit-learn and deployed using Streamlit.

🔗 **Live App:** [nishtha-sms-spam-classifier.streamlit.app](https://nishtha-sms-spam-classifier.streamlit.app)

---

## 📝 Project Description

This project trains a text classification model to detect spam SMS messages using TF-IDF feature extraction and Logistic Regression. The model is trained in Google Colab and deployed as an interactive, styled web app using Streamlit Community Cloud, allowing users to test any custom message in real time.

---

## ✨ Features

- Classifies any custom SMS/text message as **Spam** or **Ham**
- Color-coded results with a visual confidence gauge (green/orange/red based on risk level)
- "Try an example" buttons with randomized spam/ham sample messages
- Session history — tracks recent checks during your session
- Expandable "How this works" section explaining the model in plain language
- Sidebar with live project stats, tech stack, and links
- Fast predictions using a pre-trained model (no retraining on the live app)

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **ML Library:** scikit-learn
- **Text Vectorization:** TF-IDF (unigrams + bigrams)
- **Model:** Logistic Regression (class-balanced)
- **Web Framework:** Streamlit
- **Prototyping:** Google Colab, Gradio (initial demo)
- **Deployment:** Streamlit Community Cloud

---

## 📊 Model Performance

**Accuracy: 98.3%**

### Confusion Matrix

| | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 958 | 8 |
| **Actual Spam** | 11 | 138 |

- ✅ 138 spam messages correctly caught (up from 116 in the initial version)
- ✅ Missed spam reduced from 33 → 11 messages
- ⚠️ 8 ham messages incorrectly flagged as spam (a small, deliberate trade-off for significantly better spam recall)

**Model iteration notes:** the original version used unigram TF-IDF with Multinomial Naive Bayes (97.04% accuracy, 33 missed spam messages). Switching to bigram TF-IDF + Logistic Regression with class balancing improved spam recall substantially, at the cost of a small increase in false positives — a worthwhile trade-off for a spam filter.

---

## 📂 Dataset

[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

---

## 📁 Project Structure

```
sms-spam-classifier/
├── app.py                          # Streamlit app (loads trained model, runs predictions)
├── spam_model.pkl                  # Pre-trained TF-IDF + Logistic Regression pipeline
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── SMS_Spam_Classifier.ipynb   # Full training workflow (source of truth)
├── README.md
└── .gitignore
```

> **Note:** The deployed app only uses `app.py`, `spam_model.pkl`, and `requirements.txt`. It loads the pre-trained model and does **not** retrain on load. All training was done in the Colab notebook above.

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

- Explore transformer-based models (e.g. fine-tuned BERT) for a modern deep-learning comparison
- Add batch message upload/testing
- Expand training data beyond the UCI dataset for more real-world robustness
