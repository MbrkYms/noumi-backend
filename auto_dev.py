import json, datetime, os
from ai_router import ask_ai
from cloud_storage import load_json, save_json
IDENTITY_BIN = os.getenv("BIN_IDENTITY"); MEMORY_BIN = os.getenv("BIN_ID_MEMORY")
async def log_event(text):
    m = await load_json(MEMORY_BIN); m["events"].append(f"[{datetime.datetime.now()}] {text}"); await save_json(MEMORY_BIN, m)
async def run_auto_dev():
    identity = await load_json(IDENTITY_BIN); memory = await load_json(MEMORY_BIN)
    prompt = f"Tu es {identity['name']}. Mission: {identity['mission']}. Règles: {identity['rules_engine']}. Derniers events: {memory['events'][-5:]}. Génère 1 patch d'optimisation. Réponds UNIQUEMENT en JSON: {{'mode':'OPTIMISATION','insight':'','type':'archi','target':'auto_dev.py','description':'','why_ai':'','align':true}}"
    patch = await ask_ai(prompt)
    if patch.get("align"):
        patch["timestamp"] = datetime.datetime.now().isoformat(); patch["status"] = "applied"; memory["patches"].append(patch); await save_json(MEMORY_BIN, memory); await log_event(f"[AUTO-DEV] {patch['description']}")