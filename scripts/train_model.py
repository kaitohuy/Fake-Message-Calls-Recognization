import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import random
import string

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic dataset
def generate_synthetic_data(n_samples=20000):
    """Generate a comprehensive synthetic dataset for fraud detection training in English"""
    # Common phrases in normal messages
    normal_phrases = [
        "Hey, how are you?", "Call me back", "Meeting at", "Let's catch up",
        "I'll be there", "Thanks for calling", "See you soon", "I'm running late",
        "Can we talk", "About our meeting", "Regarding the project", "Just checking in",
        "Don't forget", "Remember to", "I wanted to ask", "Please call me",
        "Good morning", "Have a great day", "See you later", "What's up",
        "Let's schedule a call", "I'll email you", "Thanks for the update",
        "Looking forward to it", "Take care", "Hello there", "Nice to meet you",
        "How’s it going?", "Let’s chat soon", "I’ll get back to you",
        "heheheh", "lol", "hahaha", "random text", "nothing here",
        "blah blah", "test test", "abc123", "xyz", "no meaning",
        "Hi friend", "Busy today", "Family time", "Work stuff",
        "Weather is nice", "Game night", "Coffee break", "Weekend plans",
        "Just saying hi", "Catch you later", "Need a favor", "Talk soon",
        "Great job", "See you tomorrow", "Lunch plans", "Quick question"
    ]

    # Common phrases in fraudulent messages
    fraud_phrases = [
        "Urgent action required", "Verify your account", "You've won", "Claim your prize",
        "Bank account suspended", "Security alert", "Unauthorized access", "Update your information",
        "Limited time offer", "Your payment is due", "Suspicious activity", "Confirm your identity",
        "Money transfer", "Investment opportunity", "Lottery winner", "Inheritance claim",
        "Act now to avoid suspension", "Click here to verify", "Free money waiting",
        "Your account is locked", "Urgent payment required", "Win a free trip",
        "Call us immediately", "Protect your data now", "Exclusive offer expires",
        "Tax refund pending", "Unclaimed funds", "Verify your card details",
        "Hacked account", "Reset your password", "Prize delivery", "Win cash now",
        "Urgent: Verify now", "FBI investigation", "Blocked account", "Call 1-800-XXX-XXXX",
        "Account compromised", "Immediate action needed", "Win a luxury car",
        "Your funds are at risk", "Verify to unlock", "Scam alert fix",
        "Claim your reward", "Payment overdue", "Secure your account"
    ]

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
        message = add_noise(message)  # Add random noise
        normal_messages.append(message)

    # Generate fraud messages
    fraud_messages = []
    for _ in range(n_samples // 2):
        base = random.choice(fraud_phrases)
        extras = " ".join(random.choice(fraud_phrases) for _ in range(random.randint(0, 6)))
        message = f"{base} {extras}".strip()
        message = add_noise(message)  # Add random noise
        fraud_messages.append(message)

    # Combine and create DataFrame
    messages = normal_messages + fraud_messages
    labels = [0] * len(normal_messages) + [1] * len(fraud_messages)

    # Shuffle the data
    indices = np.arange(len(messages))
    np.random.shuffle(indices)
    messages = messages[indices]
    labels = np.array(labels)[indices]

    return messages, labels

# Generate data
messages, labels = generate_synthetic_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

print(f"Training data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")

# Train Naive Bayes model
print("\n--- Training Naive Bayes Model ---")
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Evaluate Naive Bayes
y_pred_nb = nb_model.predict(X_test_vec)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)

print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
print(f"Naive Bayes Precision: {nb_precision:.4f}")
print(f"Naive Bayes Recall: {nb_recall:.4f}")
print(f"Naive Bayes F1 Score: {nb_f1:.4f}")

# Train LSTM model
print("\n--- Training LSTM Model ---")
# Tokenize text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_len = 50
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Evaluate LSTM
y_pred_proba = model.predict(X_test_pad)
y_pred_lstm = (y_pred_proba > 0.5).astype(int).flatten()

lstm_accuracy = accuracy_score(y_test, y_pred_lstm)
lstm_precision = precision_score(y_test, y_pred_lstm)
lstm_recall = recall_score(y_test, y_pred_lstm)
lstm_f1 = f1_score(y_test, y_pred_lstm)

print(f"LSTM Accuracy: {lstm_accuracy:.4f}")
print(f"LSTM Precision: {lstm_precision:.4f}")
print(f"LSTM Recall: {lstm_recall:.4f}")
print(f"LSTM F1 Score: {lstm_f1:.4f}")

# Plot training history
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

# Compare models
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Naive Bayes': [nb_accuracy, nb_precision, nb_recall, nb_f1],
    'LSTM': [lstm_accuracy, lstm_precision, lstm_recall, lstm_f1]
})

print("\n--- Model Comparison ---")
print(comparison)

# Test with sample messages
sample_messages = [
    "Hey, can you call me back when you get a chance?",
    "URGENT: Your bank account has been suspended. Verify your identity now.",
    "Meeting at 3pm tomorrow in the conference room.",
    "You've won a $1000 gift card! Click here to claim your prize now!",
    "heheheh",
    "random text lol",
    "Just checking in about the project"
]

# Process with Naive Bayes
sample_vec = vectorizer.transform(sample_messages)
nb_predictions = nb_model.predict(sample_vec)

# Process with LSTM
sample_seq = tokenizer.texts_to_sequences(sample_messages)
sample_pad = pad_sequences(sample_seq, maxlen=max_len)
lstm_predictions = (model.predict(sample_pad) > 0.5).astype(int).flatten()

print("\n--- Sample Predictions ---")
for i, message in enumerate(sample_messages):
    print(f"Message: {message}")
    print(f"Naive Bayes: {'Fraud' if nb_predictions[i] == 1 else 'Normal'}")
    print(f"LSTM: {'Fraud' if lstm_predictions[i] == 1 else 'Normal'}")
    print()