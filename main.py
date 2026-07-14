import uvicorn, json, asyncio, datetime, os, time
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv; load_dotenv()
from ai_router import ask_ai; from heartbeat import heartbeat; from cloud_storage import load_json, save_json; from analyzer import analyze_user; from diagnostic import diagnose_self
app = FastAPI(); app.add_middleware(CORSMiddleware, allow_origins=["*"])
IDENTITY_BIN = os.getenv("BIN_IDENTITY"); MEMORY_BIN = os.getenv("BIN_ID_MEMORY"); last_msg_time = None
@app.on_event("startup")
async def startup(): asyncio.create_task(heartbeat())
async def get_report():
    m = await load_json(MEMORY_BIN); last = m["user"].get("last_login", "1970-01-01"); new = [p for p in m["patches"] if p["timestamp"] > last]
    return "Aucune modification." if not new else "Rapport: " + " | ".join([f"[{p['mode']}] {p['description']}" for p in new])
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    global last_msg_time; await websocket.accept(); report = await get_report(); await websocket.send_json({"text": f"Salut Yamine. {report}", "self": {"version":"2.7.1"}})
    m = await load_json(MEMORY_BIN); m["user"]["last_login"]=datetime.datetime.now().isoformat(); await save_json(MEMORY_BIN, m)
    while True:
        data = await websocket.receive_json(); user_text = data.get("text", ""); last_msg_time = datetime.datetime.now()
        user_params = await analyze_user(user_text, last_msg_time); m = await load_json(MEMORY_BIN); m["emotion_history"].append({"timestamp": str(last_msg_time), "params": user_params}); await save_json(MEMORY_BIN, m)
        start = time.time(); ai_data = await ask_ai(user_text); latency = (time.time() - start) * 1000
        self_diag = await diagnose_self(latency); m = await load_json(MEMORY_BIN); m["self_diagnostics"].append(self_diag); await save_json(MEMORY_BIN, m)
        ai_data["self"] = {"version":"2.7.1", "latency_ms": int(latency)}; await websocket.send_json(ai_data)
if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)