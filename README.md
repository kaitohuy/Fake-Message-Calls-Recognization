# AI Fraud Call Detection System 🛡️

An advanced AI-powered system to detect fraudulent phone calls using LSTM neural networks and natural language processing.

## Technology Stack 🛠️

**Frontend:**
- Next.js (React)
- TypeScript
- Tailwind CSS
- Shadcn UI Components

**Backend:**
- Python 3.9.11
- Flask (API server)
- TensorFlow/Keras (LSTM model)
- Scikit-learn (traditional ML models)
- NLTK (NLP processing)

## System Requirements 📋

- Node.js (v14+)
- Python (v3.9.11)
- npm (v6+)
- Git
- IDE (Recommended: PyCharm/VSCode)

## Installation Guide 🚀

### 1. Clone the Repository

```bash
git clone https://github.com/kaitohuy/Fake-Message-Calls-Recognization.git
cd Fake-Message-Calls-Recognization
```

### 2. Set Up Python Backend

Create and activate virtual environment:

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set Up Frontend

Install Node dependencies:

```bash
npm install
```

### 4. Train the LSTM Model

```bash
python scripts/train_lstm_model.py
```

This will:
- Generate synthetic training data
- Train the LSTM model
- Save model artifacts to `models/` directory
- Generate performance charts

## Running the Application ⚡

### Start Flask Backend (Terminal 1)

```bash
python scripts/flask_app.py
```
API will run at: http://localhost:5000

### Start Next.js Frontend (Terminal 2)

```bash
npm run dev
```
Frontend will run at: http://localhost:3000

## Project Structure 📂

```
├── app/                  # Next.js frontend
├── components/           # React components
├── lib/                  # Frontend utilities
├── public/               # Static assets
├── scripts/              # Python backend
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── fraud_detector.py
│   └── flask_app.py
├── models/               # Saved ML models
├── node_modules/         # Frontend dependencies
├── venv/                 # Python virtual environment
├── .env                  # Environment variables
├── README.md             # This file
└── package.json          # Node dependencies
```

