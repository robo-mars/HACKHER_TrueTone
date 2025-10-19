# api_server_fastapi.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import pickle
import logging
import uvicorn
import sqlite3
from datetime import datetime
import os
import tempfile

# Make heavy ML deps optional so the server can start in minimal environments
try:
    import whisper
    _whisper_available = True
except Exception:
    whisper = None
    _whisper_available = False
    logging.getLogger("uvicorn.error").warning("whisper not available; STT disabled")

app = FastAPI(title="Spam Call Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy model placeholders
spam_model = None
vectorizer = None
stt_model = None

def load_spam_models():
    global spam_model, vectorizer
    if spam_model is not None and vectorizer is not None:
        return
    try:
        with open('spam_detector_model.pkl', 'rb') as f:
            spam_model = pickle.load(f)
    except FileNotFoundError:
        logging.getLogger('uvicorn.error').warning('spam_detector_model.pkl not found')
        spam_model = None
    except Exception as e:
        logging.getLogger('uvicorn.error').warning(f'Failed to load spam model: {e}')
        spam_model = None

    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        logging.getLogger('uvicorn.error').warning('tfidf_vectorizer.pkl not found')
        vectorizer = None
    except Exception as e:
        logging.getLogger('uvicorn.error').warning(f'Failed to load vectorizer: {e}')
        vectorizer = None

def load_stt_model():
    global stt_model
    if stt_model is not None:
        return
    if not _whisper_available:
        logging.getLogger('uvicorn.error').warning('whisper not available; skipping STT load')
        stt_model = None
        return
    try:
        stt_model = whisper.load_model('base')
    except Exception as e:
        logging.getLogger('uvicorn.error').warning(f'Failed to load whisper model: {e}')
        stt_model = None

class CallAnalysisRequest(BaseModel):
    phone_number: str
    text: str
    caller_name: str = "Unknown"

class PredictionResponse(BaseModel):
    is_spam: bool
    confidence: float
    risk_score: int
    action: str
    message: str
    transcription: str = ""

def init_db():
    conn = sqlite3.connect("spam_calls.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS callers (
            phone_number TEXT PRIMARY KEY,
            spam_reports INTEGER DEFAULT 0,
            last_flagged TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS call_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT,
            transcription TEXT,
            is_spam BOOLEAN,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    init_db()
    load_spam_models()

@app.get("/")
def read_root():
    return {"message": "Spam Call Detection API", "status": "running"}

@app.get("/incoming-call", response_class=HTMLResponse)
async def get_call_simulator():
    try:
        # Serve the HTML file
        html_path = "incoming_call.html"
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            raise HTTPException(status_code=404, detail="HTML file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-call", response_model=PredictionResponse)
async def analyze_call(request: CallAnalysisRequest):
    load_spam_models()
    
    if spam_model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Spam detection models not available")
    
    try:
        # Predict
        text_vec = vectorizer.transform([request.text])
        prediction = spam_model.predict(text_vec)[0]
        probability = max(spam_model.predict_proba(text_vec)[0])
        
        risk_score = int(probability * 100)
        
        action = "BLOCK" if prediction == 1 and probability > 0.8 else \
                "WARN" if prediction == 1 else "ALLOW"
        
        return PredictionResponse(
            is_spam=bool(prediction),
            confidence=float(probability),
            risk_score=risk_score,
            action=action,
            message="This call appears to be spam" if prediction == 1 else "This call appears legitimate",
            transcription=request.text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-audio", response_model=PredictionResponse)
async def analyze_audio(file: UploadFile = File(...)):
    load_spam_models()
    load_stt_model()
    
    if spam_model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Spam detection models not available")
    
    if not _whisper_available or stt_model is None:
        raise HTTPException(status_code=500, detail="Speech-to-text model not available")
    
    temp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        # Transcribe audio
        result = stt_model.transcribe(temp_path)
        text = result["text"]
        
        # Analyze text
        text_vec = vectorizer.transform([text])
        prediction = spam_model.predict(text_vec)[0]
        probability = max(spam_model.predict_proba(text_vec)[0])
        
        risk_score = int(probability * 100)
        action = "BLOCK" if prediction == 1 and probability > 0.8 else \
                 "WARN" if prediction == 1 else "ALLOW"
        
        return PredictionResponse(
            is_spam=bool(prediction),
            confidence=float(probability),
            risk_score=risk_score,
            action=action,
            message="This call appears to be spam" if prediction == 1 else "This call appears legitimate",
            transcription=text
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.get("/check-caller/{phone}")
def check_caller(phone: str):
    conn = sqlite3.connect("spam_calls.db")
    c = conn.cursor()
    c.execute("SELECT spam_reports FROM callers WHERE phone_number = ?", (phone,))
    row = c.fetchone()
    conn.close()

    if row and row[0] >= 2:
        return {"is_spam": True, "reports": row[0]}
    else:
        return {"is_spam": False, "reports": row[0] if row else 0}

@app.post("/report-call")
async def report_call(data: dict):
    phone = data.get("phone_number")
    is_spam = data.get("is_spam", False)
    transcription = data.get("transcription", "")

    conn = sqlite3.connect("spam_calls.db")
    c = conn.cursor()

    # Save history
    c.execute(
        "INSERT INTO call_history (phone_number, transcription, is_spam, created_at) VALUES (?, ?, ?, ?)",
        (phone, transcription, is_spam, datetime.now().isoformat())
    )

    # Update caller reputation
    if is_spam:
        c.execute("""
            INSERT INTO callers (phone_number, spam_reports, last_flagged)
            VALUES (?, 1, ?)
            ON CONFLICT(phone_number) DO UPDATE SET
            spam_reports = spam_reports + 1,
            last_flagged = excluded.last_flagged
        """, (phone, datetime.now().isoformat()))

    conn.commit()
    conn.close()

    return {"status": "saved", "phone_number": phone}

if __name__ == "__main__":
    load_spam_models()
    uvicorn.run(app, host="0.0.0.0", port=8000)