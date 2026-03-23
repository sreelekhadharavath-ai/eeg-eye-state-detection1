# 🧠 Advanced EEG Cognitive Health Dashboard

### Live Streamlit Application for Real-Time Brainwave Monitoring
This repository contains a full end-to-end Machine Learning pipeline and interactive web dashboard for monitoring cognitive states (Focus vs. Fatigue) using an ExtraTrees Classifier mapping raw Electroencephalography (EEG) sensor data.

## 🚀 Features
- **ExtraTrees ML Model**: Trained on 14,980 instances of continuous 14-channel EEG data, achieving **95.1% Validation Accuracy**.
- **Spatial Feature Engineering**: Automatically computes cross-channel variance, standard deviation, and amplitude extremes to intelligently filter out sensor noise (e.g., sudden eye blinks).
- **Interactive Streamlit Web UI**:
  - `CSV Patient Upload`: Upload massive timeline datasets for a patient and let the model generate a chronological **Alertness Score Curve** alongside an automated medical summary.
  - `Random Simulation`: Automatically extracts isolated unseen patient records to validate the model's real-time accuracy against ground truth.
  - `Manual Entry`: Key in specific microvolt combinations directly into the 14 virtual sensors.

## 💻 Local Deployment

1. Ensure you have Python installed.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Start the local server:
```bash
streamlit run app.py
```
