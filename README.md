# AI Analysis Website 🤖✨

A modern AI-powered web application that performs **Sentiment Analysis**, **Resume Classification**, and includes an **AI Agent** for answering user queries — all in one sleek interface.

---

## 📌 Overview

This project integrates two deep learning models and an AI agent into a fully functional web application:

- 🧠 **Sentiment Analysis** — Paste any text and get Positive, Negative, or Neutral classification
- 📄 **Resume Classifier** — Upload a resume image and get Good, Average, or Poor classification
- 🤖 **AI Agent** — Ask any question and get intelligent responses powered by OpenRouter

---

## 🎨 Frontend

- **Style:** Modern dark theme with neon accents
- **Technologies:** HTML, CSS, JavaScript
- **Features:**
  - Text input for sentiment analysis
  - Image upload for resume classification
  - Chat interface for AI agent
  - Fully responsive design

---

## ⚙️ Backend

- **Framework:** FastAPI (Python)
- **Models Used:**
  - `.pt` file — Sentiment Analysis Model (Facebook BART)
  - `.h5` file — Resume Classifier Model (MobileNetV2)
- **AI Agent:** Powered by OpenRouter API
- **API Endpoints:**
  - `POST /predict-sentiment` — Returns sentiment prediction
  - `POST /classify-resume` — Returns resume quality classification
  - `POST /agent` — Returns AI agent response

---

## 🛠️ Libraries Used

| Library | Purpose |
|---|---|
| `fastapi` | Backend web framework |
| `uvicorn` | ASGI server to run FastAPI |
| `transformers` | BART sentiment model |
| `tensorflow` / `keras` | Resume classifier model |
| `pillow` | Image processing |
| `python-dotenv` | Secure API key management |
| `openrouter` | AI agent API |

---

## 📁 Project Structure

```
AI-Analysis-Website/
│
├── Backend/                        # FastAPI backend
│   ├── AI-Website-Integration.ipynb   # Integration notebook
│   └── requirements.txt               # Python dependencies
│
├── Frontend/
│   ├── index.html                     # Main web page
│   ├── style.css                      # Styling
│   └── script.js                      # Frontend logic
│
└── README.md
```

---

## 🚀 How to Run Locally

**Step 1 — Install dependencies:**
```bash
pip install fastapi uvicorn transformers tensorflow pillow python-dotenv requests
```

**Step 2 — Create `.env` file:**
```
OPENROUTER_API_KEY=your-api-key-here
```

**Step 3 — Run the backend:**
```bash
uvicorn main:app --reload
```

**Step 4 — Open `index.html`** in your browser

---

## 🌐 Deployment

| Part | Platform |
|---|---|
| Backend (FastAPI) | Render |
| Frontend (HTML/CSS/JS) | Netlify |

> **Live Demo:** Coming Soon 🚀

---

## ⚠️ Note on Model Files
Model files (`.pt` and `.h5`) are excluded from this repo due to size.

📥 Download from Google Drive: 
*[.pt file:](https://drive.google.com/file/d/1cwnGd2f0TKdJc52ZrrEnwMrz5MNThZnr/view?usp=drive_link)*
*[.h5 file:](https://drive.google.com/file/d/1Cal6rC6e8GYid4SZ3lrs244jZlTtjunf/view?usp=sharing)*

---

## 👨‍💻 Author

**Muhammad Saad Amir**
- GitHub: [MuhammadSaad19](https://github.com/MuhammadSaad19)
- LinkedIn: [muhammad-saad-amir](https://linkedin.com/in/muhammad-saad-amir)
