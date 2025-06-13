import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import random
import string

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_phrases_from_file(filepath):
    """Load phrases from a text file"""
    with open(filepath, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def generate_synthetic_data(n_samples=20000):
    # Load phrases from files in the data/ folder
    normal_phrases = load_phrases_from_file('data/normal_phrases.txt')
    fraud_phrases = load_phrases_from_file('data/fraud_phrases.txt')

    # Function to add noise (typos, special characters)
    def add_noise(text):
        if random.random() < 0.3:  # 30% chance to add noise
            text = list(text)
            for i in range(random.randint(1, 3)):
                pos = random.randint(0, len(text) - 1)
                if random.random() < 0.5:
                    text[pos] = random.choice(string.ascii_lowercase)
                else:
                    text.insert(pos, random.choice("!@#$%^&*"))
            text = ''.join(text)
        return text

    # Generate normal messages
    normal_messages = []
    for _ in range(n_samples // 2):
        base = random.choice(normal_phrases)
        extras = " ".join(random.choice(normal_phrases) for _ in range(random.randint(0, 6)))
        message = f"{base} {extras}".strip()
        message = add_noise(message)
        normal_messages.append(message)

    # Generate fraud messages
    fraud_messages = []
    for _ in range(n_samples // 2):
        base = random.choice(fraud_phrases)
        extras = " ".join(random.choice(fraud_phrases) for _ in range(random.randint(0, 6)))
        message = f"{base} {extras}".strip()
        message = add_noise(message)
        fraud_messages.append(message)

    # Combine and create DataFrame
    messages = normal_messages + fraud_messages
    labels = [0] * len(normal_messages) + [1] * len(fraud_messages)
    df = pd.DataFrame({'message': messages, 'label': labels})
    df = df.sample(frac=1).reset_index(drop=True)

    # Optional: Save to CSV for reference
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_data.csv', index=False)

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
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_len))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the LSTM model"""
    os.makedirs('models', exist_ok=True)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        'models/lstm_fraud_detection_checkpoint.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)

    return accuracy, conf_matrix, report

def save_model_and_tokenizer(model, tokenizer, max_len):
    """Save model and tokenizer"""
    os.makedirs('models', exist_ok=True)

    model.save('models/lstm_fraud_detection.h5')
    print("Model saved to models/lstm_fraud_detection.h5")

    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer saved to models/tokenizer.pickle")

    with open('models/max_len.pickle', 'wb') as handle:
        pickle.dump(max_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Max length saved to models/max_len.pickle")

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/training_history.png')
    plt.close()

    print("Training history plot saved to static/training_history.png")

def test_with_examples(model, tokenizer, max_len):
    """Test model with example messages"""
    examples = [
        "Hey, can you call me back when you get a chance?",
        "URGENT: Your bank account has been suspended. Verify your identity now.",
        "Meeting at 3pm tomorrow in the conference room.",
        "You've won a $1000 gift card! Click here to claim your prize now!",
        "heheheh",
        "random text lol",
        "Just checking in about the project",
        "Security Alert: Please reset your password."
    ]

    sequences = tokenizer.texts_to_sequences(examples)
    padded = pad_sequences(sequences, maxlen=max_len)

    predictions = model.predict(padded)

    print("\nExample Predictions:")
    for i, example in enumerate(examples):
        pred_class = "Fraud" if predictions[i][0] > 0.5 else "Normal"
        confidence = predictions[i][0] if predictions[i][0] > 0.5 else 1 - predictions[i][0]
        print(f"Message: {example}")
        print(f"Prediction: {pred_class} (Confidence: {confidence:.4f})")
        print()

def main():
    print("Generating synthetic data for training...")
    df = generate_synthetic_data(n_samples=20000)

    print("\nData Information:")
    print(f"Total samples: {len(df)}")
    print(f"Normal messages: {sum(df['label'] == 0)}")
    print(f"Fraud messages: {sum(df['label'] == 1)}")

    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, tokenizer, max_len = preprocess_data(df)

    print("\nBuilding LSTM model...")
    model = build_lstm_model(max_words=5000, max_len=max_len)

    print("\nTraining model...")
    model, history = train_model(model, X_train, y_train, X_test, y_test)

    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)

    print("\nSaving model and tokenizer...")
    save_model_and_tokenizer(model, tokenizer, max_len)

    print("\nPlotting training history...")
    plot_training_history(history)

    print("\nTesting with example messages...")
    test_with_examples(model, tokenizer, max_len)

    print("\nTraining complete!")

if __name__ == "__main__":
    main()