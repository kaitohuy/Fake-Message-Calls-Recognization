import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_or_generate_data():
    """Load existing data or generate synthetic data"""
    try:
        # Try to load existing data
        df = pd.read_csv('data/phone_messages.csv')
        print("Loaded existing dataset")
    except:
        print("Generating synthetic dataset")
        # Generate synthetic data
        normal_phrases = [
            "Hey, how are you?", "Call me back", "Meeting at", "Let's catch up",
            "I'll be there", "Thanks for calling", "See you soon", "I'm running late",
            "Can we talk", "About our meeting", "Regarding the project", "Just checking in",
            "Don't forget", "Remember to", "I wanted to ask", "Please call me"
        ]
        
        fraud_phrases = [
            "Urgent action required", "Verify your account", "You've won", "Claim your prize",
            "Bank account suspended", "Security alert", "Unauthorized access", "Update your information",
            "Limited time offer", "Your payment is due", "Suspicious activity", "Confirm your identity",
            "Money transfer", "Investment opportunity", "Lottery winner", "Inheritance claim"
        ]
        
        # Generate normal messages
        normal_messages = []
        for _ in range(2500):
            base = np.random.choice(normal_phrases)
            extras = " ".join(np.random.choice(normal_phrases, np.random.randint(0, 3)))
            normal_messages.append(f"{base} {extras}".strip())
        
        # Generate fraud messages
        fraud_messages = []
        for _ in range(2500):
            base = np.random.choice(fraud_phrases)
            extras = " ".join(np.random.choice(fraud_phrases, np.random.randint(0, 3)))
            fraud_messages.append(f"{base} {extras}".strip())
        
        # Combine and create DataFrame
        messages = normal_messages + fraud_messages
        labels = [0] * len(normal_messages) + [1] * len(fraud_messages)
        
        # Create DataFrame
        df = pd.DataFrame({
            'message': messages,
            'label': labels
        })
        
        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def preprocess_data(df):
    """Preprocess text data and prepare for LSTM model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )
    
    # Tokenize text
    max_words = 5000  # Maximum number of words to keep
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    max_len = 50  # Maximum sequence length
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    print(f"Training data shape: {X_train_pad.shape}")
    print(f"Testing data shape: {X_test_pad.shape}")
    
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, max_len

def build_lstm_model(max_words, max_len):
    """Build and compile LSTM model"""
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
    
    # LSTM layers
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    
    # Dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the LSTM model"""
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Predict on test data
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    return accuracy, conf_matrix, report

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def save_model_and_tokenizer(model, tokenizer, max_len):
    """Save model and tokenizer"""
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.save('models/lstm_fraud_detection.h5')
    
    # Save tokenizer
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save max_len
    with open('models/max_len.pickle', 'wb') as handle:
        pickle.dump(max_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Model and tokenizer saved successfully")

def test_with_examples(model, tokenizer, max_len):
    """Test model with example messages"""
    examples = [
        "Hey, can you call me back when you get a chance?",
        "URGENT: Your bank account has been suspended. Verify your identity now.",
        "Meeting at 3pm tomorrow in the conference room.",
        "You've won a $1000 gift card! Click here to claim your prize now!"
    ]
    
    # Preprocess examples
    sequences = tokenizer.texts_to_sequences(examples)
    padded = pad_sequences(sequences, maxlen=max_len)
    
    # Predict
    predictions = model.predict(padded)
    
    # Print results
    print("\nExample Predictions:")
    for i, example in enumerate(examples):
        pred_class = "Fraud" if predictions[i][0] > 0.5 else "Normal"
        confidence = predictions[i][0] if predictions[i][0] > 0.5 else 1 - predictions[i][0]
        print(f"Message: {example}")
        print(f"Prediction: {pred_class} (Confidence: {confidence:.4f})")
        print()

def main():
    # Load or generate data
    df = load_or_generate_data()
    
    # Display data info
    print("\nData Information:")
    print(f"Total samples: {len(df)}")
    print(f"Normal messages: {sum(df['label'] == 0)}")
    print(f"Fraud messages: {sum(df['label'] == 1)}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, tokenizer, max_len = preprocess_data(df)
    
    # Build model
    model = build_lstm_model(max_words=5000, max_len=max_len)
    
    # Train model
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer, max_len)
    
    # Test with examples
    test_with_examples(model, tokenizer, max_len)

if __name__ == "__main__":
    main()
