import os, re, math, time, logging
from urllib.parse import urlparse
from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from gradio_client import Client
from supabase import create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app    = FastAPI()

TRUSTED_CONTACTS = {
    "+263785544904",
    "+263776858078",
    "+263774756502",
}

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
        return {
            "label":      label,
            "score":      score,
            "latency_ms": int((time.time()-start)*1000),
        }
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
        logger.info(f"Saved to retraining queue from {user_id}")
    except Exception as e:
        logger.error(f"Retraining queue error: {e}")

def format_reply(verdict: dict) -> str:
    label = verdict["label"]
    score = verdict["score"]
    if label == "error":
        return (
            "⚠️ Service temporarily unavailable.\n"
            "Please try again in a moment.\n\n"
            "— Zim Phishing Shield"
        )
    if label == "spam":
        return (
            f"🚨 *LIKELY SPAM / PHISHING*\n"
            f"Confidence: {score:.0%}\n\n"
            f"🚫 Do NOT click any links\n"
            f"🚫 Do NOT share your OTP or PIN\n"
            f"🚫 Do NOT send money or airtime\n\n"
            f"If this message claims to be from a real institution,\n"
            f"contact them directly through their official number.\n\n"
            f"Reply *WRONG* if this verdict is incorrect.\n"
            f"— Zim Phishing Shield"
        )
    return (
        f"✅ *LIKELY LEGITIMATE*\n"
        f"Confidence: {score:.0%}\n\n"
        f"This message looks normal. Stay alert:\n"
        f"• Never share OTPs or PINs with anyone\n"
        f"• When in doubt, contact the sender directly\n\n"
        f"Reply *WRONG* if this verdict is incorrect.\n"
        f"— Zim Phishing Shield"
    )

@app.post("/whatsapp", response_class=PlainTextResponse)
async def whatsapp(Body: str = Form(...), From: str = Form(...)):
    body = Body.strip()
    logger.info(f"Message from {From}: {body[:80]}")

    if body.upper() in ("WRONG", "INCORRECT", "NO"):
        if From not in TRUSTED_CONTACTS:
            reply = (
                "🔒 Verdict corrections are restricted to authorised contacts only.\n\n"
                "— Zim Phishing Shield"
            )
        else:
            prev = last_message.get(From)
            if prev:
                old_label = prev["verdict"]["label"]
                new_label = "ham" if old_label == "spam" else "spam"
                save_for_retraining(From, prev["message"], new_label)
                log_to_db(
                    "whatsapp", From,
                    prev["message"],
                    {"label":"corrected","score":1.0},
                    feedback="incorrect"
                )
                reply = (
                    f"✅ Correction recorded.\n"
                    f"Message re-labelled as *{new_label.upper()}*.\n"
                    f"Thank you — this helps improve the model.\n\n"
                    f"— Zim Phishing Shield"
                )
            else:
                reply = (
                    "No previous message found to correct.\n"
                    "Send a message first, then reply WRONG to correct it.\n\n"
                    "— Zim Phishing Shield"
                )
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response><Message>{reply}</Message></Response>"
        )
        return PlainTextResponse(content=twiml, media_type="application/xml")

    if body.upper() in ("CORRECT", "YES", "RIGHT"):
        if From in TRUSTED_CONTACTS:
            prev = last_message.get(From)
            if prev:
                log_to_db(
                    "whatsapp", From,
                    prev["message"],
                    {"label":"confirmed","score":1.0},
                    feedback="correct"
                )
        reply = "✅ Thank you for the confirmation. Stay safe!\n\n— Zim Phishing Shield"
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response><Message>{reply}</Message></Response>"
        )
        return PlainTextResponse(content=twiml, media_type="application/xml")

    verdict = classify(body)
    log_to_db("whatsapp", From, body, verdict)
    last_message[From] = {"message": body, "verdict": verdict}
    reply = format_reply(verdict)
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{reply}</Message></Response>"
    )
    return PlainTextResponse(content=twiml, media_type="application/xml")

@app.post("/classify", response_class=JSONResponse)
async def classify_api(request: Request):
    body   = await request.json()
    text   = body.get("text", "").strip()
    source = body.get("source", "api")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    verdict = classify(text)
    log_to_db(source, None, text, verdict)
    return JSONResponse({
        "label":      verdict["label"],
        "confidence": verdict["score"],
        "latency_ms": verdict.get("latency_ms"),
    })

@app.get("/stats", response_class=JSONResponse)
async def stats():
    if not db:
        return JSONResponse({"error": "DB not configured"}, status_code=503)
    try:
        total = db.table("classifications").select("id", count="exact").execute()
        spam  = db.table("classifications").select("id", count="exact").eq("predicted_label","spam").execute()
        ham   = db.table("classifications").select("id", count="exact").eq("predicted_label","ham").execute()
        queue = db.table("retraining_queue").select("id", count="exact").execute()
        return JSONResponse({
            "total_classifications": total.count,
            "spam_detected":         spam.count,
            "ham_detected":          ham.count,
            "retraining_queue":      queue.count,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/", response_class=JSONResponse)
async def health():
    return JSONResponse({
        "status":  "ok",
        "service": "Zim Phishing Detection System",
        "model":   SPACE_NAME,
        "db":      db is not None,
    })
