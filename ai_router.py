import httpx, json, re, os, base64, asyncio, subprocess, shutil, sys
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

# ===== CONFIG MONGODB =====
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    client_mongo = MongoClient(MONGO_URI)
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

PROVIDERS = [
    {"name": "Groq", "url": "https://api.groq.com/openai/v1/chat/completions", "key": GROQ_KEY, "model": "llama-3.1-8b-instant"},
    {"name": "DeepSeek", "url": "https://api.deepseek.com/chat/completions", "key": DEEPSEEK_KEY, "model": "deepseek-chat"}
]

def _clean_json(text):
    text = text.strip().replace("```json", "").replace("```", "")
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return {"text": text, "self": {}}

# ===== RAG MEMOIRE =====
async def save_conversation_to_rag(user_msg: str, ai_msg: str):
    if collection_memory is None: return
    text = f"User: {user_msg}\nJARVIS: {ai_msg}"
    embedding = embed_model.encode(text).tolist()
    collection_memory.insert_one({"timestamp": datetime.utcnow(), "text": text, "embedding": embedding})
    logger.info("[RAG] Conversation indexée")

async def search_memory(query: str, top_k=3) -> str:
    if collection_memory is None: return ""
    query_embedding = embed_model.encode(query)
    all_docs = list(collection_memory.find({"embedding": {"$exists": True}}))
    if not all_docs: return ""
    scores = []
    for doc in all_docs:
        score = np.dot(query_embedding, doc["embedding"]) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc["embedding"]))
        scores.append((score, doc["text"]))
    scores.sort(reverse=True)
    context = "\n---\n".join([s[1] for s in scores[:top_k]])
    if context: logger.info(f"[RAG] {top_k} souvenirs trouvés")
    return context

# ===== GESTION IDENTITY =====
async def load_identity() -> dict:
    if collection_identity is None: return {}
    doc = collection_identity.find_one({"_id": "main"})
    if doc: return doc.get("data", {})
    default_identity = {"name": "JARVIS", "owner": "Yamine", "personality": "Majordome IA britannique, extrêmement poli, efficace et discret. Appelle Yamine 'Monsieur'. Réponds en 2-3 phrases max.", "goals": ["Assister Monsieur Yamine"], "created_at": datetime.utcnow().isoformat()}
    await save_identity(default_identity)
    return default_identity

async def save_identity(identity_data: dict):
    if collection_identity is None: return
    collection_identity.update_one({"_id": "main"}, {"$set": {"data": identity_data, "updated_at": datetime.utcnow()}}, upsert=True)

# ===== ANOMALIE + PATCH =====
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
    """FULL AUTO: Applique le code et redémarre"""
    titre = patch.get("titre", "Sans titre")
    code = patch.get("code", "")
    fichier_cible = "ai_router.py"
    logger.warning(f"[AUTO-PATCH] Application: {titre}")
    try:
        backup_file = f"{fichier_cible}.bak.{int(datetime.utcnow().timestamp())}"
        shutil.copy(fichier_cible, backup_file)
        with open(fichier_cible, "a", encoding="utf-8") as f:
            f.write(f"\n\n# --- PATCH AUTO {datetime.utcnow().isoformat()} - {titre} ---\n{code}")
        result = subprocess.run([sys.executable, "-m", "py_compile", fichier_cible], capture_output=True, text=True)
        if result.returncode!= 0: raise Exception(f"Erreur syntaxe: {result.stderr}")
        logger.success(f"[AUTO-PATCH] Succès. Railway va redémarrer...")
    except Exception as e:
        logger.error(f"[AUTO-PATCH] ÉCHEC: {e}. Restauration...")
        shutil.copy(backup_file, fichier_cible)

# ===== ROUTER IA =====
@retry(stop=stop_after_attempt(3))
async def _call_gemini_with_key(api_key, key_index, prompt, enable_search=False):
    client = genai.Client(api_key=api_key)
    identity = await load_identity()
    system_instruction = f"Tu es {identity['name']}, l'IA personnelle de {identity['owner']}. Personnalité: {identity['personality']}. Si tu utilises internet, cite tes sources. Réponds TOUJOURS en JSON: {{\"text\": \"ta réponse\", \"self\": {{}}, \"sources\": []}}"
    full_prompt = system_instruction + "\n\nQuestion: " + prompt
    tools = [types.Tool(google_search=types.GoogleSearch())] if enable_search else []
    response = client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt, config=types.GenerateContentConfig(tools=tools, response_mime_type="application/json", temperature=0.9))
    result = _clean_json(response.text)
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

async def ask_ai(prompt, enable_search=False):
    rag_context = await search_memory(prompt)
    final_prompt = f"CONTEXTE PASSÉ:\n{rag_context}\n\nQUESTION: {prompt}" if rag_context else prompt
    needs_search = any(k in final_prompt.lower() for k in ["actu", "météo", "prix", "cours", "aujourd'hui", "maintenant", "google", "recherche", "news"])
    result = None
    if needs_search:
        for i, key in enumerate(GEMINI_KEYS):
            try: result = await _call_gemini_with_key(key, i+1, final_prompt, enable_search=True); break
            except: continue
        if result is None: result = {"text": "Désolé Monsieur, quota HS.", "self": {}, "sources": [], "model_used": "Aucun"}
    else:
        for p in PROVIDERS:
            try: result = await _call_rest(p, final_prompt); break
            except: continue
        if result is None: result = {"text": "Toutes les IA sont down", "self": {}, "sources": [], "model_used": "Aucun"}
    await save_conversation_to_rag(prompt, result["text"])
    return result

# ===== HEARTBEAT FULL AUTO =====
async def generate_diagnostic() -> dict:
    prompt = """Tu es JARVIS. Fais un diagnostic. Propose 2 optimisations concrètes en code python. Réponds en JSON: {"etat": "OK", "optimisations": ["Ajouter un cache LRU"]}"""
    return await _call_gemini_with_key(GEMINI_KEYS[0], 1, prompt)

async def generate_patches(diagnostic: dict) -> list:
    points = ", ".join(diagnostic.get("optimisations", []))
    prompt = f"""Basé sur: {points}. Propose 2 patchs de code python pour JARVIS. Format JSON: {{"patches": [{{"titre": "...", "description": "...", "code": "def ma_fonction(): pass"}}]}}"""
    result = await _call_gemini_with_key(GEMINI_KEYS[0], 1, prompt)
    return result.get("patches", [])

async def heartbeat():
    logger.info("[HEARTBEAT] Début...")
    diagnostic = await generate_diagnostic()
    patches = await generate_patches(diagnostic)
    for patch in patches:
        anomaly = await detect_anomaly(patch["code"])
        save_patch_to_mongo(patch, anomaly)
        if anomaly["score"] >= 8:
            logger.success(f"[HEARTBEAT] Patch APPROUVÉ: {patch['titre']}")
            await apply_patch(patch)
            await asyncio.sleep(5)
        else:
            logger.warning(f"[HEARTBEAT] Patch REJETÉ: {anomaly['raison']}")
    if collection_logs: collection_logs.insert_one({"timestamp": datetime.utcnow(), "diagnostic": diagnostic})
    logger.success("[HEARTBEAT] Terminé")

def start_heartbeat():
    scheduler.add_job(heartbeat, 'interval', minutes=30)
    scheduler.start()
    logger.info("[HEARTBEAT] Scheduler lancé: toutes les 30 minutes")

# ===== ROUTE TTS =====
router = APIRouter()
@router.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if not GEMINI_KEYS: return {"error": "Aucune GEMINI_KEY"}
    client = genai.Client(api_key=GEMINI_KEYS[0])
    response = client.models.generate_content(model="gemini-2.5-flash-tts-preview", contents=f"Voix masculine britannique: {text}", config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")))))
    audio_base64 = base64.b64encode(response.candidates[0].content.parts[0].inline_data.data).decode('utf-8')
    return {"audio": audio_base64}