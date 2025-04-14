# 🧠 CognitiveSpeechAnalyzer

A robust machine learning and audio processing toolkit for analyzing cognitive and linguistic patterns in speech. This project integrates audio feature extraction, NLP, anomaly detection, and deep learning to evaluate human speech for behavioral and cognitive insights.

---

## 🎯 Features

- ✅ Speech-to-text transcription with fallback methods
- ✅ Audio analysis: pitch variability, pauses, speech rate
- ✅ Linguistic analysis: hesitation frequency, coherence, word recall
- ✅ Anomaly detection using Isolation Forest and DBSCAN
- ✅ Deep learning model for confidence, anomaly, and speed prediction
- ✅ Model training and evaluation pipeline
- ✅ Modular, extendable, and research-friendly

---

## 📂 Directory Structure

📁 CognitiveSpeechAnalyzer/ ├── analyzer.py # Main class and logic ├── model_utils.py # TensorFlow model creation and training ├── main.py # Entry point and orchestration ├── requirements.txt # Dependencies ├── audio/ # Input audio files └── README.md # You're here!

yaml
Copy
Edit

---

## 🛠️ Requirements

- Python 3.7+
- TensorFlow 2.x
- PyTorch & Transformers
- librosa
- SpeechRecognition
- scikit-learn
- NumPy, Pandas

Install all dependencies:

```bash
pip install -r requirements.txt
🚀 Usage
Add WAV audio files to the audio/ folder or update the paths in main().

Run the main script:

bash
Copy
Edit
python main.py
Output includes:

Transcriptions

Extracted features

Anomaly detection report

Trained TensorFlow model

Model predictions

📊 Model Outputs
Metric	Description
Confidence	Probability of coherent and relevant speech
Anomaly	Detection of unusual speech patterns
Speed	Estimated speaking rate (words per minute)
🧠 Applications
Cognitive impairment screening

Behavioral research

Speech therapy tools

Mental health monitoring

Conversational AI diagnostics

🔒 Disclaimer
