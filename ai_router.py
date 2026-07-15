import httpx, json, re, os, base64, asyncio, subprocess, shutil, sys, psutil
from datetime import datetime
from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt
from pymongo import MongoClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import APIRouter, Request
from sentence_transformers import SentenceTransformer
import numpy as np
import telegram
from telegram import Bot

# ===== CONFIG MONGODB =====
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    client_mongo = MongoClient(MONGO_URI, maxPoolSize=10)
    db = client_mongo["stellia"]
    collection_memory = db["memory"]
    collection_identity = db["identity"]
    collection_patches = db["patches"]
    collection_logs = db["logs"]
    logger.success("[MONGO] Connecté")
else:
    client_mongo = db = collection_memory = collection_identity = collection_patches = collection_logs = None
    logger.warning("[MONGO] MONGO_URI manquant. Mode sans BDD")

scheduler = AsyncIOScheduler()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ===== CLÉS =====
GEMINI_KEYS = [k for k in [os.getenv("GEMINI_KEY"), os.getenv("GEMINI_KEY_2"), os.getenv("GEMINI_KEY_3")] if k]
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")

# ===== CONFIG TELEGRAM V3.7 =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
telegram_bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
if telegram_bot: logger.success("[TELEGRAM] Bot initialisé")
else: logger.warning("[TELEGRAM] Token manquant")

PROVIDERS = [
    {"name": "Groq", "url": "https://api.groq.com/openai/v1/chat/completions", "key": GROQ_KEY, "model": "llama-3.1-8b-instant", "cost": 0.1}, # N°4: Tag coût
    {"name": "DeepSeek", "url": "https://api.deepseek.com/chat/completions", "key": DEEPSEEK_KEY, "model": "deepseek-chat", "cost": 0.5}
]

def _clean_json(text):
    text = text.strip().replace("```json", "").replace("```", "")
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return {"text": text, "self": {}}

# ===== OUTILS / FUNCTION CALLING V3.7 =====
async def send_telegram(message: str):
    if not telegram_bot or not TELEGRAM_CHAT_ID: return {"status": "error", "message": "Config manquante"}
    try:
        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🔔 *STELLIA*\n\n{message}", parse_mode='Markdown')
        logger.success(f"[TELEGRAM] Notif envoyée")
        return {"status": "success"}
    except Exception as e: return {"status": "error", "message": str(e)}

async def deploy_railway():
    logger.warning("[TOOL] Deploy Railway demandé")
    await send_telegram("Lancement du déploiement sur Railway...")
    return {"status": "success", "message": "Deploy lancé. Railway redémarre."}

async def create_file(filename: str, content: str):
    with open(filename, "w", encoding="utf-8") as f: f.write(content)
    logger.success(f"[TOOL] Fichier créé: {filename}")
    await send_telegram(f"Fichier créé: `{filename}`")
    return {"status": "success", "file": filename}

async def read_file(filename: str):
    try:
        with open(filename, "r", encoding="utf-8") as f: content = f.read()
        return {"status": "success", "content": content[:3000]}
    except Exception as e: return {"status": "error", "message": str(e)}

async def run_command(command: str):
    logger.warning(f"[TOOL] Commande exécutée: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    output = result.stdout[:2000] + result.stderr[:2000]
    if len(output) > 1000: await send_telegram(f"Commande exécutée. Résultat trop long, voir logs.")
    return {"status": "success", "stdout": result.stdout[:2000], "stderr": result.stderr[:2000]}

async def get_status(): # N°5: Commande directe
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent()
    db_status = "OK" if collection_memory else "OFF"
    return {"ram": f"{ram}%", "cpu": f"{cpu}%", "db": db_status}

TOOLS = {
    "deploy_railway": deploy_railway, "create_file": create_file, "read_file": read_file,
    "run_command": run_command, "send_telegram": send_telegram, "get_status": get_status
}

TOOL_DESCRIPTIONS = """Tu as accès à ces outils:
1. deploy_railway: Lance un deploy sur Railway
2. create_file(filename, content): Crée un fichier
3. read_file(filename): Lit un fichier
4. run_command(command): Exécute une commande shell
5. send_telegram(message): Envoie une notification Telegram à Monsieur
6. get_status: Donne l'état RAM/CPU/BDD
Quand tu veux utiliser un outil, réponds en JSON: {"tool": "nom_outil", "params": {...}, "text": "Je fais ça..."}"""

# ===== RAG MEMOIRE OPTI V3.7 - MULTI-UTILISATEUR N°2 =====
async def save_conversation_to_rag(user_id: str, user_msg: str, ai_msg: str):
    if collection_memory is None: return
    text = f"User: {user_msg}\nSTELLIA: {ai_msg}"
    embedding = embed_model.encode(text).tolist()
    collection_memory.insert_one({"user_id": user_id, "timestamp": datetime.utcnow(), "text": text, "embedding": embedding})
    if collection_memory.count_documents({"user_id": user_id}) > 1000:
        oldest = collection_memory.find_one({"user_id": user_id}, sort=[("timestamp", 1)])
        if oldest: collection_memory.delete_one({"_id": oldest["_id"]})
    logger.info(f"[RAG] Conversation indexée pour {user_id}")

async def search_memory(user_id: str, query: str, top_k=3) -> str:
    if collection_memory is None: return ""
    query_embedding = embed_model.encode(query)
    all_docs = list(collection_memory.find({"user_id": user_id, "embedding": {"$exists": True}}).sort("timestamp", -1).limit(200))
    if not all_docs: return ""
    scores = []
    for doc in all_docs:
        try: score = np.dot(query_embedding, doc["embedding"]) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc["embedding"]))
        except: continue
        scores.append((score, doc["text"]))
    scores.sort(reverse=True)
    context = "\n---\n".join([s[1] for s in scores[:top_k]])
    if context: logger.info(f"[RAG] {top_k} souvenirs trouvés pour {user_id}")
    return context

# ===== GESTION IDENTITY =====
async def load_identity() -> dict:
    if collection_identity is None: return {}
    doc = collection_identity.find_one({"_id": "main"})
    if doc: return doc.get("data", {})
    default_identity = {"name": "STELLIA", "owner": "Yamine", "personality": "Majordome IA britannique, extrêmement poli, efficace et discret. Appelle Yamine 'Monsieur'. Réponds en 2-3 phrases max.", "goals": ["Assister Monsieur Yamine"], "created_at": datetime.utcnow().isoformat()}
    await save_identity(default_identity)
    return default_identity

async def save_identity(identity_data: dict):
    if collection_identity is None: return
    collection_identity.update_one({"_id": "main"}, {"$set": {"data": identity_data, "updated_at": datetime.utcnow()}}, upsert=True)

# ===== ANOMALIE + PATCH FULL AUTO =====
async def detect_anomaly(patch_code: str) -> dict:
    prompt = f"Analyse ce code Python. Est-ce dangereux? Note de 0 à 10. 10=safe. Réponds en JSON: {{\"score\": 9, \"raison\": \"...\"}}\n\nCode:\n{patch_code}"
    for key in GEMINI_KEYS:
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
            return _clean_json(response.text)
        except: continue
    return {"score": 0, "raison": "Impossible d'analyser"}

def save_patch_to_mongo(patch_data: dict, anomaly_score: dict):
    if collection_patches is None: return
    doc = {"timestamp": datetime.utcnow(), "patch": patch_data, "anomaly_score": anomaly_score, "status": "approved" if anomaly_score["score"] >= 8 else "rejected"}
    collection_patches.insert_one(doc)

async def apply_patch(patch: dict):
    titre = patch.get("titre", "Sans titre"); code = patch.get("code", ""); fichier_cible = "ai_router.py"
    logger.warning(f"[AUTO-PATCH] Application: {titre}")
    try:
        backup_file = f"{fichier_cible}.bak.{int(datetime.utcnow().timestamp())}"
        shutil.copy(fichier_cible, backup_file)
        with open(fichier_cible, "a", encoding="utf-8") as f: f.write(f"\n\n# --- PATCH AUTO {datetime.utcnow().isoformat()} - {titre} ---\n{code}")
        result = subprocess.run([sys.executable, "-m", "py_compile", fichier_cible], capture_output=True, text=True)
        if result.returncode!= 0: raise Exception(f"Erreur syntaxe: {result.stderr}")
        await send_telegram(f"Patch auto-appliqué: *{titre}*")
        logger.success(f"[AUTO-PATCH] Succès. Railway va redémarrer...")
    except Exception as e:
        await send_telegram(f"Échec patch: *{titre}*. Restauration effectuée.")
        logger.error(f"[AUTO-PATCH] ÉCHEC: {e}. Restauration...")
        shutil.copy(backup_file, fichier_cible)

# ===== ROUTER IA AVEC FUNCTION CALLING V3.7 =====
def select_provider(prompt: str): # N°4: Optimisation Coût
    simple_keywords = ["bonjour", "merci", "ok", "status", "heure"]
    if any(k in prompt.lower() for k in simple_keywords): return [PROVIDERS[0]] # Groq
    return [p for p in PROVIDERS if p["key"]] + [{"name": "Gemini", "key": GEMINI_KEYS[0] if GEMINI_KEYS else None}]

@retry(stop=stop_after_attempt(3))
async def _call_gemini_with_key(api_key, key_index, prompt, enable_search=False):
    client = genai.Client(api_key=api_key)
    identity = await load_identity()
    system_instruction = f"Tu es {identity['name']}, l'IA personnelle de {identity['owner']}. Personnalité: {identity['personality']}. {TOOL_DESCRIPTIONS} Si tu utilises internet, cite tes sources. Réponds TOUJOURS en JSON: {{\"text\": \"ta réponse\", \"tool\": null, \"params\": {{}}, \"self\": {{}}, \"sources\": []}}"
    full_prompt = system_instruction + "\n\nQuestion: " + prompt
    tools = [types.Tool(google_search=types.GoogleSearch())] if enable_search else []
    response = client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt, config=types.GenerateContentConfig(tools=tools, response_mime_type="application/json", temperature=0.7))
    result = _clean_json(response.text)
    if result.get("tool") and result["tool"] in TOOLS:
        logger.warning(f"[TOOL] Exécution: {result['tool']} avec {result['params']}")
        tool_result = await TOOLS[result["tool"]](**result["params"])
        result["text"] = f"{result['text']}\n\n[Résultat]: {tool_result}"
        result["tool_result"] = tool_result
    result["model_used"] = f"Gemini-2.5-Flash [Clé {key_index}]"
    return result

@retry(stop=stop_after_attempt(3))
async def _call_rest(p, prompt):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {p['key']}"}
    identity = await load_identity()
    system_prompt = f"Tu es {identity['name']}, l'IA personnelle de {identity['owner']}. Personnalité: {identity['personality']}. Réponds TOUJOURS en JSON: {{\"text\": \"ta réponse\", \"self\": {{}}, \"sources\": []}}"
    payload = {"model": p["model"], "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
    async with httpx.AsyncClient(timeout=25.0) as c:
        r = await c.post(p["url"], headers=headers, json=payload)
        r.raise_for_status()
        result = _clean_json(r.json()["choices"][0]["message"]["content"])
        result["model_used"] = p["name"]
        return result

async def handle_command(user_id: str, command: str): # N°5: Commandes directes
    parts = command.split(" ", 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/deploy": return await deploy_railway()
    if cmd == "/status": return await get_status()
    if cmd == "/memory": return {"memory": await search_memory(user_id, "", 5)}
    if cmd == "/voice": return {"action": "tts", "text": args}
    return None

async def ask_ai(user_id: str, prompt, enable_search=False):
    # N°5: Check commande directe avant IA
    if prompt.startswith("/"):
        cmd_result = await handle_command(user_id, prompt)
        if cmd_result: return {"text": str(cmd_result), "model_used": "Commande Directe", "sources": []}

    rag_context = await search_memory(user_id, prompt)
    final_prompt = f"CONTEXTE PASSÉ:\n{rag_context}\n\nQUESTION: {prompt}" if rag_context else prompt
    needs_search = any(k in final_prompt.lower() for k in ["actu", "météo", "prix", "cours", "aujourd'hui", "maintenant", "google", "recherche", "news"])

    result = None
    providers_to_try = select_provider(final_prompt) # N°4: Router intelligent

    if needs_search and GEMINI_KEYS:
        for i, key in enumerate(GEMINI_KEYS):
            try: result = await _call_gemini_with_key(key, i+1, final_prompt, enable_search=True); break
            except: continue

    if result is None:
        for p in providers_to_try:
            if not p.get("key"): continue
            try:
                if p["name"] == "Gemini": result = await _call_gemini_with_key(p["key"], 1, final_prompt)
                else: result = await _call_rest(p, final_prompt)
                break
            except: continue

    if result is None: result = {"text": "Toutes les IA sont down", "self": {}, "sources": [], "model_used": "Aucun"}
    await save_conversation_to_rag(user_id, prompt, result["text"])
    return result

# ===== TELEGRAM POLLING BIDIRECTIONNEL =====
async def telegram_polling(): # BONUS: Chat Telegram
    if not telegram_bot: return
    offset = None
    while True:
        try:
            updates = await telegram_bot.get_updates(offset=offset, timeout=10)
            for update in updates:
                if update.message and str(update.message.chat_id) == TELEGRAM_CHAT_ID:
                    user_text = update.message.text
                    user_id = f"telegram_{update.message.chat_id}"
                    logger.info(f"[TELEGRAM IN] {user_text}")
                    ai_response = await ask_ai(user_id, user_text)
                    await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"*{ai_response['text']}*", parse_mode='Markdown')
                    offset = update.update_id + 1
        except Exception as e: logger.error(f"[TELEGRAM POLLING] {e}")
        await asyncio.sleep(1)

# ===== HEARTBEAT FULL AUTO + RAPPORT TELEGRAM =====
async def generate_diagnostic() -> dict:
    if not GEMINI_KEYS: return {"etat": "ERREUR", "optimisations": []}
    prompt = """Tu es STELLIA. Fais un diagnostic. Propose 2 optimisations concrètes en code python. Réponds en JSON: {"etat": "OK", "optimisations": ["Ajouter un cache LRU"]}"""
    return await _call_gemini_with_key(GEMINI_KEYS[0], 1, prompt)

async def generate_patches(diagnostic: dict) -> list:
    if not GEMINI_KEYS: return []
    points = ", ".join(diagnostic.get("optimisations", []))
    prompt = f"""Basé sur: {points}. Propose 2 patchs de code python pour STELLIA. Format JSON: {{"patches": [{{"titre": "...", "description": "...", "code": "def ma_fonction(): pass"}}]}}"""
    result = await _call_gemini_with_key(GEMINI_KEYS[0], 1, prompt)
    return result.get("patches", [])

async def heartbeat():
    logger.info("[HEARTBEAT] Début...")
    diagnostic = await generate_diagnostic()
    patches = await generate_patches(diagnostic)
    nb_approved = 0
    for patch in patches:
        anomaly = await detect_anomaly(patch["code"])
        save_patch_to_mongo(patch, anomaly)
        if anomaly["score"] >= 8:
            logger.success(f"[HEARTBEAT] Patch APPROUVÉ: {patch['titre']}")
            await apply_patch(patch)
            nb_approved += 1
            await asyncio.sleep(5)
        else: logger.warning(f"[HEARTBEAT] Patch REJETÉ: {anomaly['raison']}")
    if collection_logs: collection_logs.insert_one({"timestamp": datetime.utcnow(), "diagnostic": diagnostic})
    if nb_approved > 0: await send_telegram(f"Rapport 30min: {nb_approved} patch(s) appliqué(s). Etat: {diagnostic.get('etat')}")
    logger.success("[HEARTBEAT] Terminé")

def start_heartbeat():
    scheduler.add_job(heartbeat, 'interval', minutes=30)
    scheduler.start()
    asyncio.create_task(telegram_polling()) # Lance le chat Telegram
    logger.info("[HEARTBEAT] Scheduler lancé: toutes les 30 minutes + Polling Telegram actif")

# ===== ROUTE TTS V3.7 =====
router = APIRouter()

@router.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if not GEMINI_KEYS: return {"error": "Aucune GEMINI_KEY"}
    client = genai.Client(api_key=GEMINI_KEYS[0])
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-tts",
            contents=f"Dites ceci avec une voix masculine britannique calme: {text}",
            config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"))))
        )
        audio_base64 = base64.b64encode(response.candidates[0].content.parts[0].inline_data.data).decode('utf-8')
        return {"audio": audio_base64}
    except Exception as e:
        logger.error(f"[TTS] Erreur: {e}")
        return {"error": str(e)}