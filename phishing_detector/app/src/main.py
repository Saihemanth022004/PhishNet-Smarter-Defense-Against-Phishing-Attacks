
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import onnxruntime
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Load URL detection model (ONNX)
try:
    REPO_ID = "pirocheto/phishing-url-detection"
    FILENAME = "model.onnx"
    url_model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    url_detector = onnxruntime.InferenceSession(
        url_model_path,
        providers=["CPUExecutionProvider"],
    )
    print("URL detection model loaded successfully")
except Exception as e:
    print(f"Error loading URL model: {e}")
    url_detector = None

# Load email detection model (DistilBERT)
try:
    email_tokenizer = AutoTokenizer.from_pretrained("cybersectony/phishing-email-detection-distilbert_v2.4.1")
    email_model = AutoModelForSequenceClassification.from_pretrained("cybersectony/phishing-email-detection-distilbert_v2.4.1")
    print("Email detection model loaded successfully")
except Exception as e:
    print(f"Error loading email model: {e}")
    email_tokenizer = None
    email_model = None

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/predict_url', methods=['POST'])
def predict_url():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL not provided'}), 400
    
    if url_detector is None:
        return jsonify({'error': 'URL detection model not available'}), 500
    
    try:
        inputs = np.array([url], dtype="str")
        results = url_detector.run(None, {"inputs": inputs})[1]
        
        # results[0] contains probabilities for [legitimate, phishing]
        phishing_prob = results[0][1]
        legitimate_prob = results[0][0]
        
        if phishing_prob > legitimate_prob:
            label = "phishing"
            score = phishing_prob
        else:
            label = "legitimate"
            score = legitimate_prob
            
        return jsonify({'label': label, 'score': float(score)})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_email', methods=['POST'])
def predict_email():
    data = request.get_json()
    subject = data.get('subject')
    body = data.get('body')
    
    if not subject or not body:
        return jsonify({'error': 'Subject or body not provided'}), 400
    
    if email_tokenizer is None or email_model is None:
        return jsonify({'error': 'Email detection model not available'}), 500
        
    try:
        combined_text = f"Subject: {subject}\nBody: {body}"
        
        # Tokenize input
        inputs = email_tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = email_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probabilities for each class
        probs = predictions[0].tolist()
        
        # Based on the model description, it's multilabel classification
        # We'll use the highest probability to determine the result
        labels = ["legitimate_email", "phishing_url", "legitimate_url", "phishing_url_alt"]
        max_idx = probs.index(max(probs))
        
        if "phishing" in labels[max_idx]:
            label = "phishing"
        else:
            label = "not_phishing"
            
        score = max(probs)
        
        return jsonify({'label': label, 'score': float(score)})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


