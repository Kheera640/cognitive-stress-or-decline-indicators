import numpy as np
import pandas as pd

# Version sanity check
assert np.__version__ >= '1.24.0', f"Invalid NumPy version: {np.__version__}"
assert pd.__version__ >= '2.0.0', f"Invalid Pandas version: {pd.__version__}"

import librosa
import numpy as np
import pandas as pd
import re
import torch
import speech_recognition as sr
from difflib import SequenceMatcher
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
from pathlib import Path
import warnings

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class CognitiveSpeechAnalyzer:
    def __init__(self):
        self.hesitation_markers = ['uh', 'um', 'er', 'ah', 'like', 'you know']
        self.recognizer = sr.Recognizer()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.baseline_metrics = {
            'words_per_minute': 150,
            'pause_frequency': 0.5,
            'hesitation_frequency': 0.2,
            'pitch_variability': 30,
            'word_recall_score': 0.9
        }

    def transcribe_audio(self, audio_path):
        """Robust audio transcription with multiple fallbacks"""
        try:
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file {audio_path} not found")
            
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                try:
                    return self.recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    return ""
                except sr.RequestError:
                    return self.recognizer.recognize_sphinx(audio)
        except Exception as e:
            print(f"Transcription failed: {str(e)}")
            return ""

    def extract_audio_features(self, audio_path):
        """Safe audio feature extraction"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)  # Force standard sample rate
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Pause detection with noise robustness
            non_silent = librosa.effects.split(y, top_db=25, frame_length=1024, hop_length=256)
            pause_count = max(0, len(non_silent) - 1)
            
            # Pitch analysis with validation
            pitch = librosa.yin(y, fmin=80, fmax=400, sr=sr)
            valid_pitch = pitch[~np.isnan(pitch)]
            pitch_var = np.std(valid_pitch) if len(valid_pitch) > 10 else 0
            
            # Speech rate calculation
            text = self.transcribe_audio(audio_path)
            sentences = self._split_sentences(text)
            num_sentences = max(1, len(sentences))
            words = text.split()
            
            return {
                'pause_frequency': pause_count / num_sentences,
                'pitch_variability': pitch_var,
                'words_per_minute': len(words)/(duration/60) if duration > 0.1 else 0,
                'speech_duration': duration
            }
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            return None

    def analyze_linguistic_features(self, text, prompt_words=None):
        """Safe linguistic analysis with input validation"""
        if not text or len(text.strip()) == 0:
            return None
            
        try:
            sentences = self._split_sentences(text)
            words = text.split()
            word_count = len(words)
            
            # Base metrics
            metrics = {
                'hesitation_frequency': sum(text.lower().count(m) for m in self.hesitation_markers) / max(1, word_count),
                'avg_sentence_length': word_count / max(1, len(sentences)),
                'semantic_coherence': 0,
                'word_recall_score': 0,
                'word_substitutions': []
            }
            
            # Word recall analysis
            if prompt_words:
                matched = sum(1 for word in prompt_words if word.lower() in text.lower())
                metrics.update({
                    'word_recall_score': matched / len(prompt_words),
                    'word_substitutions': self._find_substitutions(text, prompt_words)
                })
            
            # Semantic coherence for valid sentences
            if len(sentences) > 1:
                metrics['semantic_coherence'] = self._calculate_coherence(sentences)
                
            return metrics
        except Exception as e:
            print(f"Linguistic analysis error: {str(e)}")
            return None

    def detect_anomalies(self, features_df):
        """Robust anomaly detection with preprocessing"""
        try:
            # Clean and scale features
            numerical_df = features_df.select_dtypes(include=[np.number]).fillna(0)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(numerical_df)
            
            # Clustering
            clustering = DBSCAN(eps=1.5, min_samples=1).fit(scaled_features)
            
            # Anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(scaled_features)
            
            return {
                'cluster': clustering.labels_,
                'anomaly_score': anomalies,
                'is_anomaly': anomalies == -1
            }
        except Exception as e:
            print(f"Anomaly detection failed: {str(e)}")
            return None

    def _split_sentences(self, text):
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def _find_substitutions(self, text, target_words):
        try:
            text_words = text.lower().split()
            return [
                (target, max(text_words, 
                    key=lambda x: SequenceMatcher(None, target.lower(), x).ratio()))
                for target in target_words
                if target.lower() not in text_words
            ]
        except:
            return []

    def _calculate_coherence(self, sentences):
        try:
            inputs = self.tokenizer(
                sentences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            embeddings = outputs.last_hidden_state[:,0,:].numpy()
            return np.mean([
                np.dot(e1, e2)/(np.linalg.norm(e1)*np.linalg.norm(e2))
                for e1, e2 in zip(embeddings[:-1], embeddings[1:])
            ])
        except:
            return 0

def main():
    analyzer = CognitiveSpeechAnalyzer()
    
    # Replace these paths with your actual WAV file locations
    audio_files = [
       r"C:\Users\abhi8\Downloads\mg.wav",
       r"C:\Users\abhi8\Downloads\rr.wav",
       r"C:\Users\abhi8\Downloads\rr2.wav",
       r"C:\Users\abhi8\Downloads\rr3.wav",
       r"C:\Users\abhi8\Downloads\rr4.wav",
       r"C:\Users\abhi8\Downloads\rr5.wav",
       r"C:\Users\abhi8\Downloads\rr6'.wav"
    ]
    
    # Validate files before processing
    valid_files = []
    for path in audio_files:
        file_path = Path(path)
        if not file_path.exists():
            print(f"Error: File not found - {file_path}")
            continue
        if file_path.suffix.lower() != '.wav':
            print(f"Error: Only WAV files supported - {file_path}")
            continue
        valid_files.append(path)
    
    if not valid_files:
        print("No valid WAV files found. Check your paths:")
        print("\n".join(audio_files))
        return

    results = []
    for path in valid_files:
        print(f"\nProcessing {Path(path).name}:")
        
        # Audio transcription with error handling
        try:
            text = analyzer.transcribe_audio(path)
            print(f"Transcription: {text[:150]}..." if text else "No speech detected")
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            continue
        
        # Feature extraction with validation
        try:
            audio_features = analyzer.extract_audio_features(path)
            ling_features = analyzer.analyze_linguistic_features(text, ["apple", "table"])
            
            if audio_features and ling_features:
                combined = {**audio_features, **ling_features}
                combined['file_name'] = Path(path).name
                results.append(combined)
            else:
                print("Feature extraction failed for this file")
        except Exception as e:
            print(f"Processing failed: {str(e)}")
            continue
    
    if results:
        df = pd.DataFrame(results)
        print("\nExtracted Features:")
        print(df[['file_name', 'words_per_minute', 'hesitation_frequency', 'word_recall_score']])
        
        # Anomaly detection
        try:
            anomalies = analyzer.detect_anomalies(df.drop(columns=['file_name'], errors='ignore'))
            if anomalies:
                df['anomaly'] = anomalies['is_anomaly']
                print("\nAnomaly Detection Results:")
                print(df[['file_name', 'words_per_minute', 'hesitation_frequency', 'anomaly']])
        except Exception as e:
            print(f"Anomaly detection failed: {str(e)}")
    else:
        print("\nNo valid results from any files")

if __name__ == "__main__":
    main()