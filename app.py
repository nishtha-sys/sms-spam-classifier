from pathlib import Path
import joblib
import streamlit as st

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

MODEL_PATH = Path(__file__).parent / "spam_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message to check whether it is likely spam.")

message = st.text_area(
    "Message",
    placeholder="Example: Congratulations! You have won a free prize. Click now!"
)

if st.button("Check message"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        spam_probability = model.predict_proba([message])[0][1]

        if prediction == 1:
            st.error(f"🚨 Spam detected\n\nSpam probability: {spam_probability:.2%}")
        else:
            st.success(f"✅ Likely not spam\n\nSpam probability: {spam_probability:.2%}")

st.caption("Built with Python, scikit-learn, and Streamlit.")