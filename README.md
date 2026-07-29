# 📩 SMS Spam Classifier

A machine learning web app that classifies SMS messages as **Spam** or **Ham (Not Spam)**, built with scikit-learn and deployed using Streamlit.

🔗 **Live App:** [nishtha-sms-spam-classifier.streamlit.app](https://nishtha-sms-spam-classifier.streamlit.app)

---

## 📝 Project Description

This project trains a text classification model to detect spam SMS messages using classic NLP techniques. The model is trained in Google Colab and deployed as an interactive web app using Streamlit Community Cloud, allowing users to test any custom message in real time.

---

## ✨ Features

- Classifies any custom SMS/text message as **Spam** or **Ham**
- Simple, interactive Streamlit web interface
- Fast predictions using a pre-trained model (no retraining on the live app)
- Clean TF-IDF + Naive Bayes pipeline

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **ML Library:** scikit-learn
- **Text Vectorization:** TF-IDF
- **Model:** Multinomial Naive Bayes
- **Web Framework:** Streamlit
- **Prototyping:** Google Colab, Gradio (initial demo)
- **Deployment:** Streamlit Community Cloud

---

## 📊 Model Performance

**Accuracy: 97.04%**

### Confusion Matrix

| | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 966 | 0 |
| **Actual Spam** | 33 | 116 |

- ✅ 0 ham messages incorrectly flagged as spam (no false positives)
- ✅ 116 spam messages correctly caught
- ⚠️ 33 spam messages missed (false negatives)

---

## 📂 Dataset

[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

---

## 📁 Project Structure

```
sms-spam-classifier/
├── app.py                          # Streamlit app (loads trained model, runs predictions)
├── spam_model.pkl                  # Pre-trained TF-IDF + Naive Bayes model
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

- Improve recall on spam detection (reduce false negatives)
- Enhance UI/UX design
- Experiment with additional models (SVM, ensemble methods)
