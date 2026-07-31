from pathlib import Path
import joblib
import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "spam_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

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
    st.metric("Model Accuracy", "98.3%")
    st.markdown("**Tech stack:** scikit-learn · TF-IDF · Logistic Regression · Streamlit")
    st.markdown("---")
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

# ---------- Example buttons ----------
st.write("**Try an example:**")
col1, col2 = st.columns(2)
example_message = None

with col1:
    if st.button("🚨 Try a spam example"):
        example_message = "Congratulations! You've won a free iPhone. Click here to claim now."

with col2:
    if st.button("✅ Try a normal example"):
        example_message = "Hey, are we still meeting for lunch tomorrow?"

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

btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    check_clicked = st.button("Check message", type="primary")
with btn_col2:
    if st.button("Clear"):
        st.session_state.message_input = ""
        st.rerun()

# ---------- Prediction ----------
if check_clicked:
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        spam_probability = model.predict_proba([message])[0][1]

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-card spam-card">
                    <div class="result-title">🚨 Spam detected</div>
                    <div>This message shows strong signs of spam content.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
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

        # Keep a simple session history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.insert(
            0,
            {
                "message": message,
                "result": "Spam" if prediction == 1 else "Ham",
                "probability": spam_probability,
            },
        )

# ---------- History ----------
if st.session_state.get("history"):
    st.markdown("---")
    st.subheader("🕒 Recent checks (this session)")
    for item in st.session_state.history[:5]:
        icon = "🚨" if item["result"] == "Spam" else "✅"
        st.write(f"{icon} **{item['result']}** ({item['probability']:.1%}) — {item['message'][:70]}{'...' if len(item['message']) > 70 else ''}")

st.markdown("---")
st.caption("Built with Python, scikit-learn, and Streamlit.")
