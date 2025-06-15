import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and stem - giữ lại từ ngữ cảnh quan trọng
    tokens = [word for word in tokens if word not in stop_words]

    # Chỉ stem các từ dài hơn 3 ký tự
    tokens = [stemmer.stem(word) if len(word) > 3 else word for word in tokens]

    # Join tokens back into text
    processed_text = ' '.join(tokens)

    return processed_text

def load_and_preprocess_data(file_path=None):
    """
    Load and preprocess data from file or generate synthetic data
    """
    if file_path:
        try:
            # Try to load data from file
            df = pd.read_csv(file_path)
            print(f"Data loaded from {file_path}")
        except:
            print(f"Could not load data from {file_path}. Generating synthetic data instead.")
            df = generate_synthetic_data()
    else:
        print("No file path provided. Generating synthetic data.")
        df = generate_synthetic_data()
    
    # Preprocess text
    df['processed_text'] = df['message'].apply(preprocess_text)
    
    return df

def generate_synthetic_data(n_samples=5000):
    """
    Generate synthetic data for demonstration
    """
    # Common phrases in normal messages
    normal_phrases = [
        "Hey, how are you?", "Call me back", "Meeting at", "Let's catch up",
        "I'll be there", "Thanks for calling", "See you soon", "I'm running late",
        "Can we talk", "About our meeting", "Regarding the project", "Just checking in",
        "Don't forget", "Remember to", "I wanted to ask", "Please call me"
    ]
    
    # Common phrases in fraudulent messages
    fraud_phrases = [
        "Urgent action required", "Verify your account", "You've won", "Claim your prize",
        "Bank account suspended", "Security alert", "Unauthorized access", "Update your information",
        "Limited time offer", "Your payment is due", "Suspicious activity", "Confirm your identity",
        "Money transfer", "Investment opportunity", "Lottery winner", "Inheritance claim"
    ]
    
    # Generate normal messages
    normal_messages = []
    for _ in range(n_samples // 2):
        base = np.random.choice(normal_phrases)
        extras = " ".join(np.random.choice(normal_phrases, np.random.randint(0, 3)))
        normal_messages.append(f"{base} {extras}".strip())
    
    # Generate fraud messages
    fraud_messages = []
    for _ in range(n_samples // 2):
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

def prepare_features(df, vectorizer_type='count'):
    """
    Prepare features for machine learning models
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Vectorize text
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=5000)
    else:
        vectorizer = CountVectorizer(max_features=5000)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Training data shape: {X_train_vec.shape}")
    print(f"Testing data shape: {X_test_vec.shape}")
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Display data info
    print("\nData Information:")
    print(f"Total samples: {len(df)}")
    print(f"Normal messages: {sum(df['label'] == 0)}")
    print(f"Fraud messages: {sum(df['label'] == 1)}")
    
    # Display sample data
    print("\nSample Data:")
    print(df.head())
    
    # Prepare features
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = prepare_features(df)
    
    print("\nFeature preparation complete. Ready for model training.")

if __name__ == "__main__":
    main()
