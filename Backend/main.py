from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import io
import os
import torch
import requests as http_requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import uvicorn
import threading
import transformers

app = FastAPI(title="AI Models API")

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 1. IMAGE MODEL ───────────────────────────────────────────────────────────
IMAGE_MODEL_PATH = "/content/drive/MyDrive/Classroom/AI 6c 4(Lab)/AI Project /AI Resume Quality Analyzer/Backend/model/Copy of resume_quality_model.h5"
image_model = load_model(IMAGE_MODEL_PATH)
IMAGE_CLASS_NAMES = ["Good", "Average", "Poor"]

# ─── 2. NLP MODEL (BART fine-tuned) ──────────────────────────────────────────
NLP_MODEL_PATH = "/content/drive/MyDrive/Classroom/AI 6c 4(Lab)/AI Project /AI Resume Quality Analyzer/Backend/model/Copy of sentiment_model_current_state.pt"
BART_MODEL_NAME = "facebook/bart-base"
NLP_CLASS_NAMES = ["Negative", "Positive", "Neutral"]

print("Loading NLP tokenizer...")
nlp_tokenizer = transformers.AutoTokenizer.from_pretrained(BART_MODEL_NAME)

print("Loading NLP model...")
nlp_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL_NAME)
nlp_model.load_state_dict(torch.load(NLP_MODEL_PATH, map_location="cpu"))
nlp_model.eval()
print("NLP model loaded ✓")

# ─── 3. OPENROUTER AGENT ─────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
chat_history = []

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
class NLPRequest(BaseModel):
    text: str

class GeminiRequest(BaseModel):
    message: str

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "AI API is running 🚀"}


# ── 1. Image Classification ───────────────────────────────────────────────────
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = image_model.predict(img_array)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        label = IMAGE_CLASS_NAMES[predicted_index] if predicted_index < len(IMAGE_CLASS_NAMES) else f"Class_{predicted_index}"

        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "all_scores": {
                IMAGE_CLASS_NAMES[i] if i < len(IMAGE_CLASS_NAMES) else f"Class_{i}": round(float(v) * 100, 2)
                for i, v in enumerate(predictions[0])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. NLP Model (BART Seq2Seq → classification via encoder logits) ───────────
@app.post("/predict/nlp")
async def predict_nlp(req: NLPRequest):
    try:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided.")

        inputs = nlp_tokenizer(
            req.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            output_ids = nlp_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                num_beams=4
            )

        # Decode generated label
        generated = nlp_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print(f"Model generated: '{generated}'")

        # Match to your class names (case-insensitive)
        generated_lower = generated.lower()
        label = "Unknown"
        for cls in NLP_CLASS_NAMES:
            if cls.lower() in generated_lower:
                label = cls
                break

        # Build confidence scores (1.0 for predicted, 0.0 for others)
        # since generation doesn't give probabilities
        all_scores = {cls: 0.0 for cls in NLP_CLASS_NAMES}
        if label != "Unknown":
            all_scores[label] = 100.0
        else:
            # fallback: show raw output
            all_scores["Raw output"] = 100.0
            label = generated  # show what model actually said

        return {
            "prediction": label,
            "confidence": 100.0 if label != "Unknown" else 0.0,
            "all_scores": all_scores
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP Error: {str(e)}")
# ── import time

MODELS_TO_TRY = [
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-coder:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

@app.post("/chat/gemini")
async def chat_gemini(req: GeminiRequest):
    global chat_history

    chat_history.append({"role": "user", "content": req.message})

    last_error = None

    for model_name in MODELS_TO_TRY:
        for attempt in range(2):  # retry once per model
            try:
                response = http_requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://localhost",
                        "X-Title": "AI Studio"
                    },
                    json={
                        "model": model_name,
                        "messages": chat_history,
                        "max_tokens": 1024
                    },
                    timeout=30
                )

                data = response.json()

                if response.status_code == 429:
                    retry_after = data.get("error", {}).get("metadata", {}).get("retry_after_seconds", 12)
                    print(f"[{model_name}] rate limited, waiting {retry_after}s...")
                    time.sleep(float(retry_after) + 1)
                    continue  # retry same model

                if response.status_code != 200:
                    last_error = data.get("error", {}).get("message", f"HTTP {response.status_code}")
                    print(f"[{model_name}] failed: {last_error}")
                    break  # try next model

                if "choices" not in data or not data["choices"]:
                    last_error = "Empty response"
                    break

                reply = data["choices"][0]["message"]["content"]
                chat_history.append({"role": "assistant", "content": reply})
                return {"reply": reply, "model_used": model_name}

            except http_requests.exceptions.Timeout:
                last_error = f"Timeout on {model_name}"
                break
            except Exception as e:
                last_error = str(e)
                break

    chat_history.pop()
    raise HTTPException(status_code=500, detail=f"All models failed. Last error: {last_error}")


# @app.post("/chat/gemini/reset")
# async def reset_gemini():
#     global chat_history
#     chat_history = []
#     return {"status": "Chat history cleared."}


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
def run_fastapi():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
t = threading.Thread(target=run_fastapi, daemon=True)
t.start()
print("FastAPI started on port 8000 ✓")