import streamlit as st
import numpy as np
import pickle

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------------------
# LOAD MODEL & SCALER
# --------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("best_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except:
        st.error("❌ Error loading model or scaler. Check file paths.")
        return None, None

model, scaler = load_model()

# --------------------------------------------------------------
# CUSTOM CSS (Sky Background + Glass UI)
# --------------------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    background: linear-gradient(135deg, #89CFF0, #A7D8FF, #F6F9FF);
    background-size: cover;
    height: 100%;
}

.glass-card {
    background: rgba(255, 255, 255, 0.30);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 35px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

.title {
    text-align: center;
    color: #084E8A;
    font-size: 42px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    margin-top: -8px;
    color: #03355A;
    font-size: 18px;
}
.predict-btn button {
    background-color: #004C99 !important;
    color: light red !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    height: 50px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About the System")
    st.write("""
    This system predicts Parkinson’s Disease based on  
    **biomedical voice measurements** using a trained Machine Learning model.
    """)
    st.markdown("---")
    st.subheader("🧠 Parkinson’s Disease")
    st.write("""
    Parkinson's is a progressive neurological condition affecting movement,  
    balance, voice & motor control.  
    Early detection helps improve treatment outcomes.
    """)
    st.markdown("---")
    st.caption("🔹Made With streamlit❤️")

# --------------------------------------------------------------
# HEADER
# --------------------------------------------------------------
st.markdown("<h1 class='title'>🧠 Parkinson's Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter the required voice-based features below</p>", unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------
# INPUT FORM (Glass Card)
# --------------------------------------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("🎤 Enter Voice Measurement Features")

col1, col2, col3 = st.columns(3)

with col1:
    fo = st.number_input("MDVP:Fo(Hz)", value=120.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", value=130.0)
    flo = st.number_input("MDVP:Flo(Hz)", value=110.0)
    jitter_percent = st.number_input("MDVP:Jitter(%)", value=0.003)
    jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.00002)
    rap = st.number_input("MDVP:RAP", value=0.001)
    ppq = st.number_input("MDVP:PPQ", value=0.001)

with col2:
    ddp = st.number_input("Jitter:DDP", value=0.004)
    shimmer = st.number_input("MDVP:Shimmer", value=0.01)
    shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.1)
    apq3 = st.number_input("Shimmer:APQ3", value=0.005)
    apq5 = st.number_input("Shimmer:APQ5", value=0.006)
    apq = st.number_input("MDVP:APQ", value=0.007)
    dda = st.number_input("Shimmer:DDA", value=0.015)

with col3:
    nhr = st.number_input("NHR", value=0.003)
    hnr = st.number_input("HNR", value=21.0)
    rpde = st.number_input("RPDE", value=0.45)
    dfa = st.number_input("DFA", value=0.7)
    spread1 = st.number_input("Spread1", value=-6.0)
    spread2 = st.number_input("Spread2", value=0.15)
    d2 = st.number_input("D2", value=2.0)
    ppe = st.number_input("PPE", value=0.1)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------------------
center_button = st.columns([2, 1, 2])[1]
with center_button:
    predict_clicked = st.button("🔮 Predict Parkinson's", use_container_width=True)

# --------------------------------------------------------------
# PREDICTION LOGIC + POPUP RECOMMENDATIONS
# --------------------------------------------------------------
if predict_clicked:

    features = np.array([
        fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
        shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
        rpde, dfa, spread1, spread2, d2, ppe
    ]).reshape(1, -1)

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error("⚠️ **High Likelihood of Parkinson’s Detected**")
        st.snow()

        st.warning("### 📌 Suggested Next Step\nConsult a neurologist for further evaluation.")

        st.info("""
        **Recommended Specialists Near You:**
        - 👨‍⚕️ Dr. R.K. Sharma – Neurologist  
        - 👩‍⚕️ Dr. Aditi Mehta – Parkinson’s Specialist  
        - 👨‍⚕️ Dr. Prakash Rao – Movement Disorder Expert  
        """)

    else:
        st.success("🎉 **No Parkinson’s Detected — You’re Doing Great!**")
        st.balloons()
        st.info("✨ Keep monitoring your health. Stay hydrated, active, and stress-free.")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------
# EXPANDER
# --------------------------------------------------------------
with st.expander("📊 Model Details"):
    st.write("""
        **Model:** Random Forest Classifier  
        **Input Features:** 22  
        **Pipeline:** Input → Scaling → Prediction  
    """)
