import os, re, math, time, logging
from urllib.parse import urlparse
from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from gradio_client import Client
from supabase import create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app    = FastAPI()

SPACE_NAME = os.getenv("SPACE_NAME", "viperDEE/spam-detector-zim")
hf_client  = None
try:
    hf_client = Client(SPACE_NAME)
    logger.info(f"Connected to HF Space: {SPACE_NAME}")
except Exception as e:
    logger.error(f"HF Space error: {e}")

db = None
try:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if url and key:
        db = create_client(url, key)
        logger.info("Connected to Supabase database.")
except Exception as e:
    logger.error(f"Supabase error: {e}")

last_message = {}

def classify(text: str) -> dict:
    start = time.time()
    try:
        result = hf_client.predict(text, api_name="/classify")
        if isinstance(result, dict) and "confidences" in result:
            top   = result["confidences"][0]
            label = top["label"]
            score = float(top["confidence"])
        elif isinstance(result, dict):
            label = max(result, key=result.get)
            score = float(result[label])
        else:
            return {"label":"error","score":0.0,"latency_ms":0}
        return {"label":label,"score":score,
                "latency_ms":int((time.time()-start)*1000)}
    except Exception as e:
        logger.error(f"Classify error: {e}")
        return {"label":"error","score":0.0,"latency_ms":0}

def log_to_db(source, user_id, message, verdict, feedback=None):
    if not db: return
    try:
        db.table("classifications").insert({
            "source":          source,
            "user_identifier": user_id,
            "message":         message,
            "message_length":  len(message),
            "predicted_label": verdict["label"],
            "confidence":      verdict["score"],
            "latency_ms":      verdict.get("latency_ms"),
            "user_feedback":   feedback,
        }).execute()
    except Exception as e:
        logger.error(f"DB log failed (ignored): {e}")

def save_for_retraining(user_id, message, correct_label):
    if not db: return
    try:
        db.table("retraining_queue").insert({
            "user_identifier": user_id,
            "message":         message,
            "correct_label":   correct_label,
            "source":          "user_feedback",
        }).execute()
    except Exception as e:
        logger.error(f"Retraining queue error: {e}")

def format_reply(verdict: dict) -> str:
    label = verdict["label"]
    score = verdict["score"]
    if label == "error":
        return "⚠️ Service temporarily unavailable. Please try again.\n\n— Zim Phishing Shield"
    if label == "spam":
        return (
            f"🚨 *LIKELY SPAM / PHISHING*\n"
            f"Confidence: {score:.0%}\n\n"
            f"🚫 Do NOT click any links\n"
            f"🚫 Do NOT share your OTP or PIN\n"
            f"🚫 Do NOT send money or airtime\n\n"
            f"Reply *WRONG* if this verdict is incorrect.\n"
            
        )
    return (
        f"✅ *LIKELY LEGITIMATE*\n"
        f"Confidence: {score:.0%}\n\n"
        f"This message looks normal. Stay alert:\n"
        f"• Never share OTPs or PINs with anyone\n"
        f"• When in doubt, contact the sender directly\n\n"
        f"Reply *WRONG* if this verdict is incorrect.\n"
        
    )

@app.post("/whatsapp", response_class=PlainTextResponse)
async def whatsapp(Body: str = Form(...), From: str = Form(...)):
    body = Body.strip()
    logger.info(f"Message from {From}: {body[:80]}")

    if body.upper() in ("WRONG","INCORRECT","NO"):
        prev = last_message.get(From)
        if prev:
            old   = prev["verdict"]["label"]
            new   = "ham" if old == "spam" else "spam"
            save_for_retraining(From, prev["message"], new)
            reply = (
                f"✅ Thank you! Recorded as *{new.upper()}*.\n"
                f"This helps improve our model.\n\n— Zim Phishing Shield"
            )
        else:
            reply = "No previous message found to correct."
        twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{reply}</Message></Response>'
        return PlainTextResponse(content=twiml, media_type="application/xml")

    if body.upper() in ("CORRECT","YES","RIGHT"):
        log_to_db("whatsapp", From,
                  last_message.get(From, {}).get("message",""),
                  {"label":"confirmed","score":1.0}, feedback="correct")
        reply = "✅ Great! Glad the verdict was accurate. Stay safe!"
        twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{reply}</Message></Response>'
        return PlainTextResponse(content=twiml, media_type="application/xml")

    verdict = classify(body)
    log_to_db("whatsapp", From, body, verdict)
    last_message[From] = {"message": body, "verdict": verdict}
    reply  = format_reply(verdict)
    twiml  = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{reply}</Message></Response>'
    return PlainTextResponse(content=twiml, media_type="application/xml")

@app.post("/classify", response_class=JSONResponse)
async def classify_api(request: Request):
    body   = await request.json()
    text   = body.get("text","").strip()
    source = body.get("source","api")
    if not text:
        return JSONResponse({"error":"No text"}, status_code=400)
    verdict = classify(text)
    log_to_db(source, None, text, verdict)
    return JSONResponse({
        "label":      verdict["label"],
        "confidence": verdict["score"],
        "latency_ms": verdict.get("latency_ms"),
    })

@app.get("/", response_class=JSONResponse)
async def health():
    return JSONResponse({
        "status":  "ok",
        "service": "Zim Phishing Detection System",
        "model":   SPACE_NAME,
        "db":      db is not None,
    })
