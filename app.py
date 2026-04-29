import os
from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from gradio_client import Client

app = FastAPI()

SPACE_NAME = os.getenv("SPACE_NAME", "viperDEE/spam-detector-zim")
client = Client(SPACE_NAME)


def classify(text: str) -> dict:
    try:
        result = client.predict(text, api_name="/classify")
        if isinstance(result, dict) and "confidences" in result:
            top = result["confidences"][0]
            return {"label": top["label"], "score": float(top["confidence"])}
        if isinstance(result, dict):
            top_label = max(result, key=result.get)
            return {"label": top_label, "score": float(result[top_label])}
        return {"label": "error", "score": 0.0}
    except Exception as e:
        return {"label": "error", "score": 0.0, "detail": str(e)[:200]}


def format_reply(result: dict) -> str:
    label = result["label"]
    score = result["score"]

    if label == "error":
        return (
            "Service temporarily unavailable. Please try again shortly.\n\n"
            "- Zim Spam Detector"
        )

    if label == "spam":
        return (
            f"⚠️ LIKELY SPAM / PHISHING\n"
            f"Confidence: {score:.0%}\n\n"
            f"Do NOT click any links\n"
            f"Do NOT share OTPs or PINs\n"
            f"Do NOT send money\n\n"
            f"If the message claims to be from a real institution, "
            f"contact them directly using a number from their official website.\n\n"
            f"- Zim Spam Detector"
        )

    return (
        f"✅ LIKELY LEGITIMATE\n"
        f"Confidence: {score:.0%}\n\n"
        f"This message looks normal, but stay alert. "
        f"When in doubt, verify directly with the sender.\n\n"
        f"- Zim Spam Detector"
    )


@app.post("/whatsapp")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    result = classify(Body)
    reply = format_reply(result)
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{reply}</Message></Response>"
    )
    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.get("/")
async def health():
    return {"status": "ok", "service": "Zim Spam Detector", "space": SPACE_NAME}