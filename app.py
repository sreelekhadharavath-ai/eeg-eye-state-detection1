import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# ==========================================
# 1. UI Design & Page Configuration
# ==========================================
st.set_page_config(page_title="Advanced EEG Cognitive Health Dashboard", page_icon="🧠", layout="wide")

st.title("🧠 Advanced EEG Cognitive Health Dashboard")
st.markdown("---")

# ==========================================
# 2. Data Handling & Model Loading
# ==========================================
def load_components():
    try:
        model = joblib.load("rf_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please run `eeg_classifier.py` first to train and save the model.")
        st.stop()

model, scaler = load_components()

def add_engineered_features(data):
    features = data.copy()
    channel_cols = features.columns[:14]
    features['mean_amp'] = features[channel_cols].mean(axis=1)
    features['std_amp'] = features[channel_cols].std(axis=1)
    features['max_amp'] = features[channel_cols].max(axis=1)
    features['min_amp'] = features[channel_cols].min(axis=1)
    return features

# ==========================================
# 3. Smart Output Conversion Logic
# ==========================================
def generate_health_insights(prediction, probability):
    alertness = probability[0] * 100 
    if prediction == 0:  
        if alertness > 85:
            return "Hyper-Focused", "High", "Low", "Optimum Health / Normal", alertness
        else:
            return "Relaxed / Distracted", "Medium", "Moderate", "Mild Stress / Starting to Lose Focus", alertness
    else:  
        if alertness < 15:
            return "Deeply Drowsy / Asleep", "Critically Low", "High", "Severe Fatigue Risk! Action Required.", alertness
        else:
            return "Drowsy / Micro-sleeps", "Low", "Moderate-High", "Fatigue Warning! Take a break.", alertness

# ==========================================
# 4. Input Options & User Experience
# ==========================================
st.sidebar.header("⬇️ Get the Data")
try:
    with open("eeg_data.csv", "rb") as file:
        st.sidebar.download_button(
            label="Download Full EEG Dataset (CSV)",
            data=file,
            file_name="OpenML_EEG_EyeState.csv",
            mime="text/csv",
            help="Click to download the 14,980 rows of OpenML training data directly to your laptop's Downloads folder."
        )
except:
    pass

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Input Options")
mode = st.sidebar.radio("Select how to input EEG Data:", ["Upload Patient CSV Data", "Random Sample from Dataset", "Manual Entry"])

st.sidebar.markdown("---")
st.sidebar.info("This professional dashboard maps raw 14-Channel EEG sensor data through an ExtraTrees Machine Learning model to evaluate cognitive health & fatigue in real-time.")

input_df = None
sample_idx = None
actual_label = None

if mode == "Random Sample from Dataset":
    st.subheader("📊 Simulate Real Patient Data")
    st.write("Extracts a random 14-channel EEG reading from the dataset to simulate a real-time hospital scan.")
    try:
        df = pd.read_csv("eeg_data.csv")
        features = df.drop(columns=['eye_state'])
        target = df['eye_state']
        
        if st.button("Extract Random Sample & Analyze", type="primary"):
            sample_idx = np.random.choice(len(df))
            input_df = features.iloc[[sample_idx]].copy()
            actual_label = target.iloc[sample_idx]
            st.success(f"Successfully extracted Patient Record #{sample_idx}")
    except FileNotFoundError:
        st.error("Dataset `eeg_data.csv` not found! Please ensure it is downloaded.")

elif mode == "Manual Entry":
    st.subheader("⌨️ Manual Data Entry")
    st.write("Input the raw microvolt readings from the 14 EEG nodes.")
    with st.form("manual_entry_form"):
        col1, col2 = st.columns(2)
        user_inputs = []
        for i in range(1, 15):
            with col1 if i <= 7 else col2:
                val = st.number_input(f"EEG Node {i} (V{i})", value=4200.0, step=10.0)
                user_inputs.append(val)
        
        submitted = st.form_submit_button("Predict Cognitive State", type="primary")
        if submitted:
            try:
                df_header = pd.read_csv("eeg_data.csv", nrows=0)
                feature_cols = df_header.columns[:-1]
            except:
                feature_cols = [f"V{i}" for i in range(1, 15)]
            
            input_array = np.array(user_inputs).reshape(1, -1)
            input_df = pd.DataFrame(input_array, columns=feature_cols)

elif mode == "Upload Patient CSV Data":
    st.subheader("📂 Upload Patient EEG Data")
    st.write("Upload a CSV file containing 14-channel EEG recordings for a patient over a session.")
    
    # Download sample template button (Optional but good UX)
    st.download_button(
        label="Download Example CSV Template",
        data="V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14\n4300,4000,4200,4100,4300,4600,4000,4600,4200,4200,4100,4200,4600,4300\n",
        file_name="patient_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Choose a Patient Document (CSV or TXT)", type=["csv", "txt"])
    
    if uploaded_file is not None:
        try:
            patient_df = pd.read_csv(uploaded_file)
            
            # Automatically drop any text columns (like timestamps or patient names)
            patient_df = patient_df.select_dtypes(include=[np.number])
            
            if len(patient_df.columns) < 14:
                st.error(f"❌ **Model Limitation:** Your document only has {len(patient_df.columns)} numeric columns/channels. This Machine Learning model was trained on a 14-channel headset, meaning it mathematically requires at least 14 streams of EEG data to make a prediction.")
            else:
                st.success(f"Successfully loaded patient session with {len(patient_df)} chronological readings!")
                
                # Automatically grab the first 14 numeric columns, regardless of what they are named
                features_only = patient_df.iloc[:, :14].copy()
                # Standardize column names secretly so the scaler doesn't crash
                features_only.columns = [f"V{i}" for i in range(1, 15)]
                
                with st.spinner("Running full session cognitive analysis..."):
                    time.sleep(1)
                    
                    patient_eng = add_engineered_features(features_only)
                    scaled_patient = scaler.transform(patient_eng)
                    
                    predictions = model.predict(scaled_patient)
                    probabilities = model.predict_proba(scaled_patient)
                    
                    alertness_scores = probabilities[:, 0] * 100
                    percent_drowsy = (sum(predictions == 1) / len(predictions)) * 100
                    avg_alertness = np.mean(alertness_scores)
                    
                    st.markdown("---")
                    st.subheader("📋 Full Session Cognitive Overview")
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.info(f"**Total Time Steps Analyzed:**\n### {len(predictions)}")
                    with c2:
                        color = "green" if avg_alertness > 60 else "red"
                        st.markdown(f"**Average Alertness Score:**\n### <span style='color:{color}'>{avg_alertness:.1f}%</span>", unsafe_allow_html=True)
                    with c3:
                        if percent_drowsy > 30:
                            st.error(f"**Time Spent Drowsy:**\n### {percent_drowsy:.1f}%")
                        else:
                            st.success(f"**Time Spent Drowsy:**\n### {percent_drowsy:.1f}%")
                            
                    st.markdown("---")
                    st.subheader("⏱️ Alertness Score Timeline")
                    st.write("Tracking the patient's focus and fatigue levels dynamically across the entire recording session. (100% = Hyper Focused, 0% = Asleep)")
                    
                    chart_data = pd.DataFrame(alertness_scores, columns=["Alertness Score (%)"])
                    st.line_chart(chart_data)
                    
                    st.markdown("### 👨‍⚕️ Automated Doctor's Note")
                    if avg_alertness > 80:
                        st.success("The patient maintained excellent, sustained focus throughout the monitored testing session.")
                    elif percent_drowsy > 50:
                        st.error("CRITICAL FATIGUE DETECTED. The patient spent the absolute majority of this session in a medically compromising drowsy state.")
                    else:
                        st.warning("The patient showed signs of heavily fluctuating attention spans and mild cognitive fatigue.")
                        
        except Exception as e:
            st.error(f"Error processing the file: {e}")

# ==========================================
# 5. Prediction & Displaying Results (Single Row Data)
# ==========================================
if input_df is not None:
    with st.spinner("Analyzing brainwave patterns through ML model..."):
        time.sleep(1)
        
        input_eng = add_engineered_features(input_df)
        scaled_data = scaler.transform(input_eng)
        
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]
        
    cog_state, attention, fatigue, insight, alertness = generate_health_insights(prediction, probabilities)
    
    st.markdown("---")
    st.subheader("💡 Cognitive Health Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 0:
            st.success(f"**Cognitive State:**\n\n### {cog_state}")
        else:
            st.error(f"**Cognitive State:**\n\n### {cog_state}")
            
    with col2:
        st.info(f"**Attention Level:**\n\n### {attention}")
        
    with col3:
        if fatigue == "Low":
            st.success(f"**Fatigue Level:**\n\n### {fatigue}")
        elif fatigue == "High":
            st.error(f"**Fatigue Level:**\n\n### {fatigue}")
        else:
            st.warning(f"**Fatigue Level:**\n\n### {fatigue}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    insight_container = st.container(border=True)
    with insight_container:
        st.markdown(f"**🩺 Health Insight:** {insight}")
        st.markdown(f"**⚡ Alertness Score:** {alertness:.1f}%")
        st.progress(int(alertness))
        
        if actual_label is not None:
            st.markdown("---")
            truth_str = "Focused (Eyes Open)" if actual_label == 0 else "Drowsy (Eyes Closed)"
            pred_str = "Focused (Eyes Open)" if prediction == 0 else "Drowsy (Eyes Closed)"
            
            if prediction == actual_label:
                st.write(f"✅ **Model Validation:** Correct! Ground truth was also **{truth_str}**.")
            else:
                st.write(f"❌ **Model Validation:** Incorrect. Ground truth was **{truth_str}**, but model predicted **{pred_str}**.")

    st.markdown("---")
    st.subheader("📈 Raw EEG Signal Trace")
    st.write("Visualizing the 14-channel electrical trace for the captured moment in time.")
    
    raw_14_channels = input_df.iloc[0].values
    chart_data = pd.DataFrame(raw_14_channels, columns=["Microvolts"], index=input_df.columns)
    st.line_chart(chart_data)
