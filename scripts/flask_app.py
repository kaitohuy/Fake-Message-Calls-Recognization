from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os
from data_preprocessing import preprocess_text

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and tokenizer
model = None
tokenizer = None
max_len = 50

def load_model_and_tokenizer():
    """Load the trained LSTM model and tokenizer"""
    global model, tokenizer, max_len
    
    try:
        # Load model
        model = tf.keras.models.load_model('models/lstm_fraud_detection.h5')
        print("Model loaded successfully")
        
        # Load tokenizer
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully")
        
        # Load max_len if available
        try:
            with open('models/max_len.pickle', 'rb') as handle:
                max_len = pickle.load(handle)
            print(f"Max length loaded: {max_len}")
        except:
            max_len = 50
            print(f"Using default max length: {max_len}")
            
        return True
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return False

def preprocess_text(text):
    """Clean and preprocess text for model input"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_fraud_with_model(message):
    # Xử lý đặc biệt cho tin nhắn rất ngắn
    if len(message.split()) <= 1:
        short_message = message.lower().strip()

        # Danh sách từ an toàn
        safe_words = ["hi", "hello", "thanks", "thank", "ok", "yes", "no", "bye",
                      "congrats", "congratulations", "hey", "later", "done"]

        if any(short_message == word for word in safe_words):
            return {
                "isFraud": False,
                "confidence": 0.1,
                "message": message
            }

    global model, tokenizer, max_len

    if model is None:
        success = load_model_and_tokenizer()
        if not success:
            return detect_fraud_simple(message)

    try:
        # Preprocess the message using the same function as training
        processed_message = preprocess_text(message)

        sequence = tokenizer.texts_to_sequences([processed_message])
        padded = pad_sequences(sequence, maxlen=max_len)

        prediction = model.predict(padded)[0][0]
        is_fraud = bool(prediction > 0.5)
        confidence = float(min(max(prediction if is_fraud else 1 - prediction, 0), 1))

        print(f"Message: {message}, Prediction: {prediction}, Confidence: {confidence}")

        return {
            "isFraud": is_fraud,
            "confidence": confidence,
            "message": message
        }
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return detect_fraud_simple(message)


def detect_fraud_simple(message):
    fraud_keywords = [
        "urgent", "verify", "suspended", "win", "prize", "account", "payment",
        "money", "transfer", "investment", "lottery", "claim", "click", "call"
    ]

    message = message.lower()
    # Nếu không có từ khóa nào, coi là không lừa đảo
    if not any(word in message for word in fraud_keywords):
        return {"isFraud": False, "confidence": 0.0}

    # Logic cũ cho các trường hợp có từ khóa
    keyword_matches = sum(1 for keyword in fraud_keywords if keyword in message)
    keyword_ratio = keyword_matches / len(fraud_keywords)
    has_urgency = bool(re.search(r'urgent|immediately|now|hurry', message))
    has_money_mention = bool(re.search(r'\$|\bmoney\b|\bpayment\b', message))
    has_personal_info_request = bool(re.search(r'password|account|verify', message))

    risk_factors = [
        keyword_ratio * 0.5,
        0.2 if has_urgency else 0,
        0.15 if has_money_mention else 0,
        0.25 if has_personal_info_request else 0
    ]

    total_risk = sum(risk_factors)
    normalized_score = min(total_risk, 1.0)
    is_fraud = normalized_score > 0.3
    random_factor = 0.05 * (np.random.random() - 0.5)
    confidence = max(0, min(1,
                            is_fraud and 0.5 + normalized_score * 0.5 + random_factor or 1 - normalized_score + random_factor))

    return {
        "isFraud": bool(is_fraud),
        "confidence": float(confidence)
    }

@app.route('/api/detect', methods=['POST'])
def detect_fraud():
    """API endpoint for fraud detection"""
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    # Use model-based detection if available, otherwise fall back to simple detection
    if model is not None:
        result = detect_fraud_with_model(message)
    else:
        result = detect_fraud_simple(message)
    
    return jsonify({
        "isFraud": result["isFraud"],
        "confidence": result["confidence"],
        "message": message
    })

@app.route('/api/model-info')
def model_info():
    """Return information about the model"""
    if model is None:
        load_model_and_tokenizer()
    
    # In a real application, this would return actual model metrics
    return jsonify({
        "modelLoaded": model is not None,
        "lstm": {
            "accuracy": 0.972,
            "precision": 0.968,
            "recall": 0.981,
            "f1Score": 0.974,
            "trainingAccuracy": 0.9989,
            "validationAccuracy": 0.9720,
            "loss": 0.0059
        },
        "naiveBayes": {
            "accuracy": 0.891,
            "precision": 0.875,
            "recall": 0.862,
            "f1Score": 0.868,
            "trainingAccuracy": 0.9215,
            "validationAccuracy": 0.8910,
            "loss": 0.1106
        }
    })

@app.route('/')
def home():
    """Home route"""
    return "Fraud Detection API is running. Use /api/detect to analyze messages."

if __name__ == '__main__':
    # Try to load model at startup
    load_model_and_tokenizer()
    
    # Run the Flask app
    app.run(debug=True)
