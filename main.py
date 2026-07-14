import uvicorn, json, asyncio, datetime, os, time
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv; load_dotenv()

from ai_router import ask_ai
from heartbeat import heartbeat
from cloud_storage import load_json, save_json
from analyzer import analyze_user
from diagnostic import diagnose_self

import httpx # AJOUT POUR TTS

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

IDENTITY_BIN = os.getenv("BIN_IDENTITY")
MEMORY_BIN = os.getenv("BIN_ID_MEMORY")
TTS_API_KEY = os.getenv("TTS_API_KEY") # AJOUTE DANS RAILWAY

last_msg_time = None

@app.on_event("startup")
async def startup():
    asyncio.create_task(heartbeat())

async def get_report():
    m = await load_json(MEMORY_BIN)
    last = m["user"].get("last_login", "1970-01-01")
    new = [p for p in m["patches"] if p["timestamp"] > last]
    return "Aucune modification." if not new else "Rapport: " + " | ".join([f"[{p['mode']}] {p['description']}" for p in new])

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    global last_msg_time
    await websocket.accept()
    
    report = await get_report()
    await websocket.send_json({"response": f"Salut Yamine. {report}", "self": {"version":"2.7.1"}}) # FIX: "response" au lieu de "text"

    m = await load_json(MEMORY_BIN)
    m["user"]["last_login"]=datetime.datetime.now().isoformat()
    await save_json(MEMORY_BIN, m)

    while True:
        data = await websocket.receive_json()
        user_text = data.get("user_input", "") # FIX: "user_input" pour matcher le front
        last_msg_time = datetime.datetime.now()

        user_params = await analyze_user(user_text, last_msg_time)
        m = await load_json(MEMORY_BIN)
        m["emotion_history"].append({"timestamp": str(last_msg_time), "params": user_params})
        await save_json(MEMORY_BIN, m)

        start = time.time()
        ai_data = await ask_ai(user_text)
        latency = (time.time() - start) * 1000

        self_diag = await diagnose_self(latency)
        m = await load_json(MEMORY_BIN)
        m["self_diagnostics"].append(self_diag)
        await save_json(MEMORY_BIN, m)

        ai_data["self"] = {"version":"2.7.1", "latency_ms": int(latency)}
        ai_data["response"] = ai_data.get("text", "Erreur") # FIX: force la clé "response"
        await websocket.send_json(ai_data)

# ===== NOUVEAU : ROUTE TTS GOOGLE =====
@app.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if not TTS_API_KEY:
        return JSONResponse({"error": "TTS_API_KEY manquante"}, status_code=500)

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={TTS_API_KEY}"
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": "fr-FR", "name": "fr-FR-Wavenet-C"}, # Voix JARVIS Homme
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.1}
    }

    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.post(url, json=payload)
        r.raise_for_status()
        result = r.json()

    return {"audio": result['audioContent']} # base64

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)