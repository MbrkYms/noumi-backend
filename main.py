import uvicorn, json, asyncio, datetime, os, time
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv; load_dotenv()
from ai_router import ask_ai, start_heartbeat # <-- ON IMPORTE LE NOUVEAU HEARTBEAT
from analyzer import analyze_user
from diagnostic import diagnose_self
# NOUVEAUX IMPORTS
from pymongo import MongoClient
from loguru import logger

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ===== CONFIG MONGODB =====
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    client_mongo = MongoClient(MONGO_URI)
    db = client_mongo["stellia"]
    collection_memory = db["memory"] # Remplace IDENTITY_BIN + MEMORY_BIN
    logger.success("[MONGO] Client initialisé")
else:
    client_mongo = db = collection_memory = None
    logger.warning("[MONGO] MONGO_URI manquant. Mode sans BDD")

TTS_API_KEY = os.getenv("TTS_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
PORT = int(os.getenv("PORT", 8080))
last_msg_time = None

# ===== FONCTION POUR REMPLACER load_json/save_json =====
async def get_memory():
    if collection_memory is None: # <-- FIX 1: is None au lieu de not
        return { "user_id": "yamine", "user": {"last_login": "1970-01-01"}, "emotion_history": [], "self_diagnostics": [], "patches": [] }
    mem = collection_memory.find_one({"user_id": "yamine"})
    if not mem:
        mem = {
            "user_id": "yamine",
            "user": {"last_login": "1970-01-01"},
            "emotion_history": [],
            "self_diagnostics": [],
            "patches": []
        }
        collection_memory.insert_one(mem)
        logger.info("[MEMORY] Nouveau document créé dans Mongo")
    return mem

async def save_memory(mem):
    if collection_memory is not None: # <-- FIX 2: is not None au lieu de if
        collection_memory.update_one({"user_id": "yamine"}, {"$set": mem}, upsert=True)

@app.on_event("startup")
async def startup():
    start_heartbeat() # <-- LANCE LE HEARTBEAT DE AI_ROUTER
    logger.info("STELLIA V3.3.2 Démarrée")

async def get_report():
    m = await get_memory()
    last = m["user"].get("last_login", "1970-01-01")
    new = [p for p in m.get("patches", []) if p.get("timestamp", "") > last] # FIX: .get()
    return "Aucune modification." if not new else "Rapport: " + " | ".join([f"[{p.get('status','OPTIMISATION')}] {p['patch']['titre']}" for p in new])

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    global last_msg_time
    await websocket.accept()
    report = await get_report()
    await websocket.send_json({"response": f"Salut Yamine. Je suis Stellia. {report}", "self": {"version":"3.3.2"}})
    m = await get_memory()
    m["user"]["last_login"] = datetime.datetime.now().isoformat() # FIX: datetime.datetime
    await save_memory(m)
    while True:
        data = await websocket.receive_json()
        user_text = data.get("user_input", "")
        last_msg_time = datetime.datetime.now() # FIX: datetime.datetime
        user_params = await analyze_user(user_text, last_msg_time)
        m = await get_memory()
        m["emotion_history"].append({"timestamp": str(last_msg_time), "params": user_params})
        await save_memory(m)
        start = time.time()
        ai_data = await ask_ai(user_text, enable_search=True) # ACTIVE LA RECHERCHE
        latency = (time.time() - start) * 1000
        self_diag = await diagnose_self(latency)
        m = await get_memory()
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