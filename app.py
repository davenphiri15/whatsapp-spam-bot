import os
import time
import logging
from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from gradio_client import Client
from supabase import create_client, Client as SupabaseClient

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Zim Phishing Detection System",
    description="WhatsApp chatbot backend for phishing message detection.",
    version="1.0.0",
)

# ── Hugging Face Space (model API) ─────────────────────────────────────────
SPACE_NAME = os.getenv("SPACE_NAME", "viperDEE/spam-detector-zim")
hf_client: Client | None = None
try:
    hf_client = Client(SPACE_NAME)
    logger.info(f"Connected to HF Space: {SPACE_NAME}")
except Exception as e:
    logger.error(f"Failed to connect to HF Space: {e}")

# ── Supabase ───────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
db: SupabaseClient | None = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        db = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Connected to Supabase database.")
    else:
        logger.warning("Supabase credentials not set — logging disabled.")
except Exception as e:
    logger.error(f"Supabase connection failed: {e}")


# ── Classification ─────────────────────────────────────────────────────────
def classify(text: str) -> dict:
    """Send text to the HF Space and return the top prediction."""
    start = time.time()
    try:
        if not hf_client:
            return {"label": "error", "score": 0.0, "latency_ms": 0}

        result = hf_client.predict(text, api_name="/classify")

        if isinstance(result, dict) and "confidences" in result:
            top = result["confidences"][0]
            label = top["label"]
            score = float(top["confidence"])
        elif isinstance(result, dict):
            label = max(result, key=result.get)
            score = float(result[label])
        else:
            return {"label": "error", "score": 0.0, "latency_ms": 0}

        return {
            "label": label,
            "score": score,
            "latency_ms": int((time.time() - start) * 1000),
        }

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"label": "error", "score": 0.0, "latency_ms": 0}


# ── Database logging ───────────────────────────────────────────────────────
def log_classification(
    source: str,
    user_id: str | None,
    message: str,
    verdict: dict,
) -> None:
    """Log a classification result to Supabase. Fails silently."""
    if not db:
        return
    try:
        db.table("classifications").insert({
            "source": source,
            "user_identifier": user_id,
            "message": message,
            "message_length": len(message),
            "predicted_label": verdict["label"],
            "confidence": verdict["score"],
            "latency_ms": verdict.get("latency_ms"),
        }).execute()
    except Exception as e:
        logger.error(f"DB log failed (ignored): {e}")


# ── Reply formatting ───────────────────────────────────────────────────────
def format_whatsapp_reply(verdict: dict) -> str:
    """Format the classification verdict as a WhatsApp-friendly message."""
    label = verdict["label"]
    score = verdict["score"]

    if label == "error":
        return (
            "⚠️ Service temporarily unavailable.\n\n"
            "We could not classify your message right now. "
            "Please try again in a moment.\n\n"
            
        )

    if label == "spam":
        return (
            f"🚨 *LIKELY SPAM / PHISHING*\n"
            f"Confidence: {score:.0%}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🚫 Do NOT click any links\n"
            f"🚫 Do NOT share your OTP or PIN\n"
            f"🚫 Do NOT send money or airtime\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"If the message appears to be from a real institution "
            f"such as EcoCash, a bank, or ZIMRA, contact them directly "
            f"using the number on their official website — "
            f"not the number in this message.\n\n"
            
        )

    return (
        f"✅ *LIKELY LEGITIMATE*\n"
        f"Confidence: {score:.0%}\n\n"
        f"This message appears normal. However, always stay alert:\n"
        f"• Never share OTPs or PINs with anyone\n"
        f"• When in doubt, contact the sender directly\n"
        f"• Scammers constantly improve their techniques\n\n"
        
    )


# ── Routes ─────────────────────────────────────────────────────────────────
@app.post("/whatsapp", response_class=PlainTextResponse)
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    """
    Receives incoming WhatsApp messages from Twilio.
    Classifies the message, logs to DB, and returns a TwiML reply.
    """
    logger.info(f"Message from {From}: {Body[:80]}...")

    verdict = classify(Body)
    log_classification(source="whatsapp", user_id=From, message=Body, verdict=verdict)
    reply = format_whatsapp_reply(verdict)

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{reply}</Message></Response>"
    )
    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.post("/classify", response_class=JSONResponse)
async def classify_api(request: Request):
    """
    Direct classification API for the browser plugin.
    Accepts JSON: {"text": "...", "source": "browser_plugin"}
    Returns JSON: {"label": "spam"|"ham", "confidence": 0.99}
    """
    body = await request.json()
    text = body.get("text", "").strip()
    source = body.get("source", "api")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    verdict = classify(text)
    log_classification(source=source, user_id=None, message=text, verdict=verdict)

    return JSONResponse({
        "label": verdict["label"],
        "confidence": verdict["score"],
        "latency_ms": verdict.get("latency_ms"),
    })


@app.get("/stats", response_class=JSONResponse)
async def stats():
    """Returns basic usage statistics from the database."""
    if not db:
        return JSONResponse({"error": "Database not configured"}, status_code=503)
    try:
        total = db.table("classifications").select("id", count="exact").execute()
        spam  = db.table("classifications").select("id", count="exact").eq("predicted_label", "spam").execute()
        ham   = db.table("classifications").select("id", count="exact").eq("predicted_label", "ham").execute()
        wa    = db.table("classifications").select("id", count="exact").eq("source", "whatsapp").execute()
        plugin = db.table("classifications").select("id", count="exact").eq("source", "browser_plugin").execute()
        return JSONResponse({
            "total_classifications": total.count,
            "spam_detected":         spam.count,
            "ham_detected":          ham.count,
            "by_source": {
                "whatsapp":       wa.count,
                "browser_plugin": plugin.count,
            },
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=JSONResponse)
async def health():
    """Health check endpoint. Used by UptimeRobot to keep the service warm."""
    return JSONResponse({
        "status":       "ok",
        "service":      "Zim Phishing Detection System",
        "model_space":  SPACE_NAME,
        "db_connected": db is not None,
    })
