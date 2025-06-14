
---

```markdown
# 🤖 AI Fraud Call Detection System

A full-stack AI-powered web application to detect fraudulent phone call messages using LSTM neural networks and NLP techniques.

## 🧠 Features

- Detect fraud messages with a trained LSTM model
- Real-time analysis via Flask API
- Interactive web interface (built with Next.js + Tailwind CSS)
- Synthetic message generation and history tracking
- Simple fallback rule-based detection if model is unavailable

---

## 🛠️ System Requirements

- [Node.js](https://nodejs.org/) (v14 or higher)
- [Python](https://www.python.org/) **v3.9.11**
- [npm](https://www.npmjs.com/) (v6 or higher)
- [Git](https://git-scm.com/)
- IDE: **PyCharm** (recommended)

---

## 📁 Project Structure

```

├── app/                  # Next.js frontend
├── components/           # React components
├── lib/                  # Frontend utilities (API calls, etc.)
├── public/               # Static assets (logo, icons, etc.)
├── scripts/              # Python backend and ML scripts
│   ├── flask\_app.py
│   ├── train\_lstm\_model.py
│   └── data\_preprocessing.py
├── models/               # Saved model and tokenizer (after training)
├── node\_modules/         # Frontend dependencies
├── venv/                 # Python virtual environment
├── .env                  # Environment variables
├── package.json          # Node dependencies
├── tailwind.config.ts    # Tailwind CSS config
└── tsconfig.json         # TypeScript config

````

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kaitohuy/Fake-Message-Calls-Recognization.git
cd Fake-Message-Calls-Recognization
````

---

### 2️⃣ Set Up Python Backend

#### 2.1 Create a Virtual Environment

```bash
python -m venv venv
```

Activate environment:

```bash
.\venv\Scripts\activate
```

> 🔁 *Run this again every time you reopen the terminal*

#### 2.2 Install Python Dependencies

```bash
pip install flask flask-cors tensorflow numpy pandas scikit-learn matplotlib
```

---

### 3️⃣ Set Up Frontend (Next.js)

```bash
npm install
```

---

### 4️⃣ Train the LSTM Model

```bash
python scripts/train_lstm_model.py
```

This will:

* Generate synthetic data
* Train the LSTM model
* Save the model + tokenizer to `models/`
* Export accuracy/loss charts to `static/`

---

### 5️⃣ Run the Application

#### 5.1 Start Flask Backend – Terminal 1

```bash
python scripts/flask_app.py
```

> 📍 Runs at: [http://localhost:5000](http://localhost:5000)

#### 5.2 Start Next.js Frontend – Terminal 2

```bash
npm run dev
```

> 📍 Runs at: [http://localhost:3000](http://localhost:3000)



## 📦 Built With

* 🧠 **TensorFlow/Keras** – LSTM model
* 🧪 **scikit-learn** – Naive Bayes fallback
* 🧪 **Flask** – Python API backend
* ⚡ **Next.js** – Frontend framework
* 🎨 **Tailwind CSS** – UI styling
* ❤️ **React** – Frontend components

---


> 💡 Maintained by [@kaitohuy](https://github.com/kaitohuy)
