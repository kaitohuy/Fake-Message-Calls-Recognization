Here's a comprehensive `README.md` file for your GitHub repository:

```markdown
# AI Fraud Call Detection System 🛡️

An advanced AI-powered system to detect fraudulent phone calls using LSTM neural networks and natural language processing.

![System Demo](demo.gif) <!-- Add a demo GIF if available -->

## Features ✨

- Real-time fraud detection for phone call transcripts
- Multi-model ensemble (LSTM, Random Forest, SVM, Naive Bayes)
- Rule-based pattern matching
- Confidence scoring and risk factor analysis
- Interactive dashboard with visualizations
- API endpoint for integration

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

## API Endpoints 🌐

`POST /api/analyze`
- Request body: `{ "message": "your call transcript here" }`
- Response:
```json
{
  "is_fraud": true,
  "confidence": 85.5,
  "risk_factors": ["urgent language", "phone number detected"]
}
```

## Contributing 🤝

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Research paper: "Artificial Intelligence Based Fake or Fraud Phone Calls Detection"
- NLTK and TensorFlow communities
- Shadcn UI component library
```

This README includes:

1. Project title and brief description
2. Key features
3. Technology stack
4. System requirements
5. Step-by-step installation guide
6. Running instructions
7. Project structure
8. API documentation
9. Contribution guidelines
10. License information
11. Acknowledgments

You can customize it further by:
- Adding a demo GIF/video
- Including screenshots
- Adding badges (build status, license, etc.)
- Expanding the API documentation
- Adding troubleshooting section
