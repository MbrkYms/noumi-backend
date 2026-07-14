import uvicorn, json, asyncio, os, time
import datetime # <-- FIX: import normal
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv; load_dotenv()

from ai_router import ask_ai, start_heartbeat, load_memory, save_memory, load_identity, save_identity # <-- ON IMPORTE LA MEMOIRE MONGO
from analyzer import analyze_user
from diagnostic import diagnose_self
from loguru import logger

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TTS_API_KEY = os.getenv("TTS_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
PORT = int(os.getenv("PORT", 8080))
last_msg_time = None

# ===== PLUS BESOIN DE get_memory/save_memory ICI =====
# On utilise directement celles de ai_router

@app.on_event("startup")
async def startup():
    start_heartbeat() # <-- LANCE LE HEARTBEAT DE AI_ROUTER
    logger.info("STELLIA V3.3.2 Démarrée - Mode MongoDB")

async def get_report():
    m = await load_memory() # <-- UTILISE LA FONCTION MONGO
    last = m.get("user", {}).get("last_login", "1970-01-01")
    patches = await load_identity() # On peut stocker les patches dans identity
    new = [p for p in patches.get("patches", []) if p.get("timestamp", "") > last]
    return "Aucune modification." if not new else "Rapport: " + " | ".join([f"[{p.get('status','OPTIMISATION')}] {p['patch']['titre']}" for p in new])

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    global last_msg_time
    await websocket.accept()
    
    report = await get_report()
    await websocket.send_json({"response": f"Salut Yamine. Je suis Stellia. {report}", "self": {"version":"3.3.2"}})
    
    m = await load_memory()
    if "user" not in m: m["user"] = {}
    m["user"]["last_login"] = datetime.datetime.now().isoformat() # <-- FIX ICI
    await save_memory(m)
    
    while True:
        data = await websocket.receive_json()
        user_text = data.get("user_input", "")
        last_msg_time = datetime.datetime.now() # <-- FIX ICI
        
        user_params = await analyze_user(user_text, last_msg_time)
        
        m = await load_memory()
        if "emotion_history" not in m: m["emotion_history"] = []
        m["emotion_history"].append({"timestamp": str(last_msg_time), "params": user_params})
        await save_memory(m)
        
        start = time.time()
        ai_data = await ask_ai(user_text, enable_search=True) # ACTIVE LA RECHERCHE
        latency = (time.time() - start) * 1000
        
        self_diag = await diagnose_self(latency)
        
        m = await load_memory()
        if "self_diagnostics" not in m: m["self_diagnostics"] = []
        m["self_diagnostics"].append(self_diag)
        await save_memory(m)
        
        # FIX: On ajoute les sources + model_used
        ai_data["self"] = {
            "version": "3.3.2",
            "latency_ms": int(latency),
            "sources": ai_data.get("sources", [])
        }
        ai_data["response"] = ai_data.get("text", "Je n'ai pas compris")
        
        await websocket.send_json(ai_data)

# ===== ROUTE TTS GEMINI - REMPLACE GOOGLE TTS =====
from ai_router import router as tts_router
app.include_router(tts_router) # <-- ON UTILISE LE TTS DE AI_ROUTER

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)