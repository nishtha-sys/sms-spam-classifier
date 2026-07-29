# 📩 SMS Spam Classifier

A beginner-friendly machine learning project that classifies SMS messages as **Spam** or **Not Spam (Ham)**.

🔗 **Live App:** [Open the SMS Spam Classifier](https://nishtha-sms-spam-classifier.streamlit.app)

## Overview

This project uses Natural Language Processing (NLP) and machine learning to identify potentially unwanted SMS messages.

The model converts text into numerical features using **TF-IDF Vectorization** and predicts the label using a **Multinomial Naive Bayes** classifier.

## Features

- Predicts whether an SMS message is spam or not spam
- Shows the spam probability for each message
- Interactive web interface built with Streamlit
- Model training and evaluation in Google Colab
- Publicly deployed web application

## Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 97.04% |
| Correctly identified normal messages | 966 |
| Normal messages incorrectly marked as spam | 0 |
| Correctly identified spam messages | 116 |
| Spam messages missed by the model | 33 |

## Tech Stack

- Python
- Pandas
- scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Streamlit
- Google Colab
- GitHub

## Dataset

The project uses the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), containing 5,574 labelled SMS messages.

## Project Structure

```text
sms-spam-classifier/
├── app.py                 # Streamlit web application
├── train.py               # Model-training script
├── spam_model.pkl         # Saved trained model
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── notebooks/
    └── SMS_Spam_Classifier.ipynb
