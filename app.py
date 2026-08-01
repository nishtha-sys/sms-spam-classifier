from pathlib import Path
import joblib
import streamlit as st
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- Page config ----------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "spam_model.pkl"
HF_MODEL_NAME = "nishthasahani/sms-spam-distilbert"


@st.cache_resource
def load_sklearn_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    model.eval()
    return tokenizer, model


def predict_bert(message, tokenizer, model):
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    spam_prob = probs[1].item()
    prediction = 1 if spam_prob > 0.5 else 0
    return prediction, spam_prob


def get_top_spam_words(message, sklearn_pipeline, top_n=5):
    """Return the words/phrases in this message that most pushed the prediction toward spam."""
    vectorizer = sklearn_pipeline.named_steps["vectorizer"]
    classifier = sklearn_pipeline.named_steps["classifier"]

    features = vectorizer.transform([message])
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    nonzero_indices = features.nonzero()[1]
    contributions = [
        (feature_names[i], features[0, i] * coefficients[i])
        for i in nonzero_indices
    ]
    contributions.sort(key=lambda x: x[1], reverse=True)
    return [word for word, score in contributions[:top_n] if score > 0]


sklearn_model = load_sklearn_model()

# ---------- Custom styling ----------
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }
    .stTextArea textarea {
        font-size: 1rem;
        border-radius: 10px;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 0;
        background-color: #4F46E5;
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #4338CA;
        color: white;
    }
    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .spam-card {
        background-color: #DC2626;
        border: 1px solid #B91C1C;
        color: #FFFFFF;
    }
    .ham-card {
        background-color: #16A34A;
        border: 1px solid #15803D;
        color: #FFFFFF;
    }
    .uncertain-card {
        background-color: #CA8A04;
        border: 1px solid #A16207;
        color: #FFFFFF;
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .gauge-track {
        width: 100%;
        height: 14px;
        background-color: #E5E7EB;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About this project")
    st.write(
        "A machine learning model that classifies SMS messages as "
        "**Spam** or **Ham (Not Spam)**."
    )
    st.markdown("---")
    st.subheader("🧠 Choose a model")
    model_choice = st.radio(
        "Model",
        ["⚡ Fast (Logistic Regression)", "🎯 Accurate (DistilBERT)"],
        label_visibility="collapsed",
    )
    if "DistilBERT" in model_choice:
        st.metric("Model Accuracy", "99.0%")
        st.caption("Transformer-based, higher accuracy, slightly slower")
    else:
        st.metric("Model Accuracy", "98.3%")
        st.caption("TF-IDF + Logistic Regression, instant predictions")
    st.markdown("---")
    st.markdown("**Tech stack:** scikit-learn · Transformers · Streamlit")
    st.markdown("[📂 View source on GitHub](https://github.com/nishtha-sys/sms-spam-classifier)")
    st.markdown("**Dataset:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)")

# ---------- Header ----------
st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message below to check whether it's likely spam.")

with st.expander("ℹ️ How this works"):
    st.markdown(
        """
        This app uses a **TF-IDF vectorizer** to convert your message into numeric features
        based on word patterns (including two-word phrases like *"click here"* or *"free entry"*),
        then a **Logistic Regression** classifier predicts whether those patterns match spam
        or genuine messages.

        - **Trained on:** UCI SMS Spam Collection (5,574 labeled messages)
        - **Test accuracy:** 98.3%
        - The model outputs a probability, not a certainty — borderline messages may be
          harder to classify, just like for a human reader.
        """
    )

SPAM_EXAMPLES = [
    "Congratulations! You've won a free iPhone. Click here to claim now.",
    "URGENT! Your mobile number has won £2000 in our prize draw. Call 09061701461 now.",
    "FREE entry to win a brand new iPhone. Text WIN to 80086 now, limited slots!",
    "You have been selected for a cash loan of $5000. No credit check needed. Reply YES.",
    "Your account will be suspended. Verify your details immediately at this link.",
    "Congrats! You've been selected to receive a free cruise vacation. Call now to claim your spot.",
]

HAM_EXAMPLES = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Don't forget to submit the assignment before 5 PM today.",
    "Can you send me the notes from today's class?",
    "Mom, I'll be home late tonight, don't wait for dinner.",
    "Meeting rescheduled to 3 PM, please update your calendar.",
    "Happy birthday! Hope you have an amazing day",
]

# ---------- Example buttons ----------
st.write("**Try an example:**")
col1, col2 = st.columns(2)
example_message = None

with col1:
    if st.button("🚨 Try a spam example"):
        example_message = random.choice(SPAM_EXAMPLES)

with col2:
    if st.button("✅ Try a normal example"):
        example_message = random.choice(HAM_EXAMPLES)

# ---------- Session state for message box ----------
if "message_input" not in st.session_state:
    st.session_state.message_input = ""

if example_message:
    st.session_state.message_input = example_message

# ---------- Input ----------
message = st.text_area(
    "Message",
    key="message_input",
    placeholder="Example: Congratulations! You have won a free prize. Click now!",
    height=120,
)

def clear_message():
    st.session_state.message_input = ""


btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    check_clicked = st.button("Check message", type="primary")
with btn_col2:
    st.button("Clear", on_click=clear_message)

# ---------- Prediction ----------
if check_clicked:
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        if "DistilBERT" in model_choice:
            with st.spinner("Loading DistilBERT model (first run takes a moment)..."):
                tokenizer, bert_model = load_bert_model()
            prediction, spam_probability = predict_bert(message, tokenizer, bert_model)
        else:
            prediction = sklearn_model.predict([message])[0]
            spam_probability = sklearn_model.predict_proba([message])[0][1]

        if 0.45 <= spam_probability <= 0.55:
            st.markdown(
                f"""
                <div class="result-card uncertain-card">
                    <div class="result-title">🤔 Uncertain — needs review</div>
                    <div>This message is borderline. The model isn't confident either way — consider reviewing it manually.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif prediction == 1:
            st.markdown(
                f"""
                <div class="result-card spam-card">
                    <div class="result-title">🚨 Spam detected</div>
                    <div>This message shows strong signs of spam content.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "DistilBERT" not in model_choice:
                top_words = get_top_spam_words(message, sklearn_model)
                if top_words:
                    st.caption("🔍 Words that most influenced this result: " + ", ".join(f"**{w}**" for w in top_words))
        else:
            st.markdown(
                f"""
                <div class="result-card ham-card">
                    <div class="result-title">✅ Likely not spam</div>
                    <div>This message looks like a normal, genuine message.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        pct = float(spam_probability) * 100
        if pct < 30:
            gauge_color = "#22C55E"  # green — low risk
        elif pct < 70:
            gauge_color = "#F59E0B"  # orange — medium risk
        else:
            gauge_color = "#EF4444"  # red — high risk

        st.write("**Spam probability**")
        st.markdown(
            f"""
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{pct}%; background-color:{gauge_color};"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"{spam_probability:.2%} likelihood of being spam")

        # Keep a simple session history (skip if identical to the last entry)
        if "history" not in st.session_state:
            st.session_state.history = []

        current_model_label = "DistilBERT" if "DistilBERT" in model_choice else "Logistic Regression"
        if 0.45 <= spam_probability <= 0.55:
            result_label = "Uncertain"
        else:
            result_label = "Spam" if prediction == 1 else "Ham"

        is_duplicate = (
            st.session_state.history
            and st.session_state.history[0]["message"] == message
            and st.session_state.history[0]["model"] == current_model_label
        )
        if not is_duplicate:
            st.session_state.history.insert(
                0,
                {
                    "message": message,
                    "result": result_label,
                    "probability": spam_probability,
                    "model": current_model_label,
                },
            )

# ---------- History ----------
if st.session_state.get("history"):
    st.markdown("---")
    st.subheader("🕒 Recent checks (this session)")
    for item in st.session_state.history[:5]:
        if item["result"] == "Spam":
            icon = "🚨"
        elif item["result"] == "Ham":
            icon = "✅"
        else:
            icon = "🤔"
        st.write(f"{icon} **{item['result']}** ({item['probability']:.1%}) — *{item['model']}* — {item['message'][:60]}{'...' if len(item['message']) > 60 else ''}")

st.markdown("---")
st.subheader("📂 Batch check (upload a CSV or TXT file)")
st.caption("CSV should have one column of messages (any header name). TXT should have one message per line.")

uploaded_file = st.file_uploader("Upload file", type=["csv", "txt"], label_visibility="collapsed")

if uploaded_file is not None:
    import pandas as pd
    import io

    if uploaded_file.name.endswith(".csv"):
        batch_df = pd.read_csv(uploaded_file)
        message_col = batch_df.columns[0]
        messages_list = batch_df[message_col].astype(str).tolist()
    else:
        content = uploaded_file.read().decode("utf-8")
        messages_list = [line.strip() for line in content.splitlines() if line.strip()]

    st.write(f"Found **{len(messages_list)}** messages.")

    if st.button("Classify all messages"):
        with st.spinner(f"Classifying {len(messages_list)} messages..."):
            results = []
            use_bert = "DistilBERT" in model_choice
            if use_bert:
                tokenizer, bert_model = load_bert_model()

            for msg in messages_list:
                if use_bert:
                    pred, prob = predict_bert(msg, tokenizer, bert_model)
                else:
                    pred = sklearn_model.predict([msg])[0]
                    prob = sklearn_model.predict_proba([msg])[0][1]

                if 0.45 <= prob <= 0.55:
                    label = "Uncertain"
                else:
                    label = "Spam" if pred == 1 else "Ham"

                results.append({"message": msg, "prediction": label, "spam_probability": round(prob, 4)})

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_buffer.getvalue(),
                file_name="spam_classification_results.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("Built with Python, scikit-learn, and Streamlit.")
