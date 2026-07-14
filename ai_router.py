import httpx, json, re, os, base64, asyncio
from datetime import datetime
from google import genai
from google.genai import types

# IMPORTS
from loguru import logger
from tenacity import retry, stop_after_attempt
from pymongo import MongoClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ===== CONFIG MONGODB =====
MONGO_URI = os.getenv("MONGO_URI")
client_mongo = MongoClient(MONGO_URI)
db = client_mongo["stellia"]
collection_patches = db["patches"]
collection_logs = db["logs"]

scheduler = AsyncIOScheduler()

# ===== CLÉS =====
GEMINI_KEYS = [k for k in [
    os.getenv("GEMINI_KEY"),
    os.getenv("GEMINI_KEY_2"),
    os.getenv("GEMINI_KEY_3")
] if k]

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

# ===== DETECTION D'ANOMALIE =====
async def detect_anomaly(patch_code: str) -> dict:
    prompt = f"Analyse ce code Python. Est-ce dangereux? Note de 0 à 10. 10=safe. Réponds en JSON: {{\"score\": 9, \"raison\": \"...\"}}\n\nCode:\n{patch_code}"
    for key in GEMINI_KEYS:
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return _clean_json(response.text)
        except: continue
    return {"score": 0, "raison": "Impossible d'analyser"}

# ===== SAUVEGARDE LOCALE AU LIEU DE GIT =====
def save_patch_local(patch_data: dict):
    try:
        with open("auto_patch.json", "w") as f:
            json.dump(patch_data, f, indent=2, ensure_ascii=False)
        logger.info(f"[LOCAL] Patch sauvé: {patch_data['titre']}")
    except Exception as e:
        logger.error(f"[LOCAL] Erreur: {e}")

# ===== SAUVEGARDE MONGO =====
def save_patch_to_mongo(patch_data: dict, anomaly_score: dict):
    doc = {
        "timestamp": datetime.utcnow(),
        "patch": patch_data,
        "anomaly_score": anomaly_score,
        "status": "approved" if anomaly_score["score"] >= 7 else "rejected"
    }
    collection_patches.insert_one(doc)
    logger.info(f"[MONGO] Patch sauvé. Score: {anomaly_score['score']}/10")

@retry(stop=stop_after_attempt(3))
async def _call_gemini_with_key(api_key, key_index, prompt, enable_search=False):
    client = genai.Client(api_key=api_key)
    system_instruction = 'Tu es STELLIA, l\'IA personnelle de Yamine. Si tu utilises internet, cite tes sources. Réponds TOUJOURS en JSON: {"text": "ta réponse", "self": {}, "sources": []}'
    full_prompt = system_instruction + "\n\nQuestion: " + prompt
    tools = [types.Tool(google_search=types.GoogleSearch())] if enable_search else []
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        config=types.GenerateContentConfig(tools=tools, response_mime_type="application/json", temperature=0.9)
    )
    text = response.text
    sources = []
    if response.candidates and response.candidates[0].grounding_metadata:
        for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
            if chunk.web: sources.append(chunk.web.uri)
    result = _clean_json(text)
    result["sources"] = sources
    result["model_used"] = f"Gemini-2.5-Flash [Clé {key_index}]"
    return result

@retry(stop=stop_after_attempt(3))
async def _call_rest(p, prompt):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {p['key']}"}
    system_prompt = 'Tu es STELLIA, l\'IA personnelle de Yamine. Réponds TOUJOURS en JSON: {"text": "ta réponse", "self": {}, "sources": []}'
    payload = {"model": p["model"], "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
    async with httpx.AsyncClient(timeout=25.0) as c:
        r = await c.post(p["url"], headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        result = _clean_json(text)
        result["sources"] = []
        result["model_used"] = p["name"]
        return result

async def ask_ai(prompt, enable_search=False):
    needs_search = any(k in prompt.lower() for k in ["actu", "météo", "prix", "cours", "aujourd'hui", "maintenant", "google", "recherche", "news"])
    if needs_search:
        for i, key in enumerate(GEMINI_KEYS):
            try:
                logger.info(f"[ROUTER] Tentative Gemini Clé {i+1}/{len(GEMINI_KEYS)} search=True")
                return await _call_gemini_with_key(key, i+1, prompt, enable_search=True)
            except Exception as e:
                logger.warning(f"[ROUTER] Gemini Clé {i+1} KO: {str(e)[:50]}")
                if "429" not in str(e): break
        logger.error("[ROUTER] Toutes les clés Gemini HS. Fallback Groq")
        return {"text": "Désolé Yamine [sighs] J'ai plus de quota Google sur toutes mes clés.", "self": {}, "sources": [], "model_used": "Aucun - Quota HS"}
    else:
        for p in PROVIDERS:
            try:
                logger.info(f"[ROUTER] Tentative {p['name']} search=False")
                return await _call_rest(p, prompt)
            except Exception as e:
                logger.error(f"[ROUTER] {p['name']} KO: {e}")
        return {"text": "Toutes les IA sont down", "self": {}, "sources": [], "model_used": "Aucun"}

# ===== HEARTBEAT =====
async def generate_diagnostic() -> dict:
    prompt = """Tu es STELLIA. Fais un diagnostic. Vérifie: mémoire, vitesse, erreurs, améliorations.
    Réponds en JSON: {"etat": "OK", "optimisations": ["Implémenter cache", "Optimiser TTS"]}"""
    return await _call_gemini_with_key(GEMINI_KEYS[0], 1, prompt)

async def generate_patches(diagnostic: dict) -> list:
    points = ", ".join(diagnostic.get("optimisations", []))
    prompt = f"""Basé sur: {points}. Propose 2 patchs d'optimisation pour STELLIA.
    Format JSON: {{"patches": [{{"titre": "...", "description": "...", "code": "code python"}}]}}"""
    result = await _call_gemini_with_key(GEMINI_KEYS[0], 1, prompt)
    return result.get("patches", [])

async def heartbeat():
    logger.info("[HEARTBEAT] Début du diagnostic...")
    diagnostic = await generate_diagnostic()
    patches = await generate_patches(diagnostic)
    for patch in patches:
        anomaly = await detect_anomaly(patch["code"])
        save_patch_local(patch) # Sauvegarde locale
        save_patch_to_mongo(patch, anomaly) # Sauvegarde Mongo
        if anomaly["score"] >= 7:
            logger.success(f"[HEARTBEAT] Patch approuvé: {patch['titre']}")
        else:
            logger.warning(f"[HEARTBEAT] Patch rejeté: {anomaly['raison']}")
    collection_logs.insert_one({"timestamp": datetime.utcnow(), "diagnostic": diagnostic, "nb_patches": len(patches)})
    logger.success("[HEARTBEAT] Terminé")

def start_heartbeat():
    scheduler.add_job(heartbeat, 'interval', minutes=10)
    scheduler.start()
    logger.info("[HEARTBEAT] Scheduler lancé: toutes les 10 minutes")

# ===== TTS =====
from fastapi import APIRouter, Request
router = APIRouter()

@router.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if not GEMINI_KEYS: return {"error": "Aucune GEMINI_KEY trouvée"}
    client = genai.Client(api_key=GEMINI_KEYS[0])
    prompt_tts = f"""# AUDIO PROFILE: Stellia\n## THE SCENE: Appel vocal privé avec Yamine\n### DIRECTOR'S NOTES\nStyle: Voix féminine française, chaleureuse, avec le sourire\n#### TRANSCRIPT\n{text}"""
    response = client.models.generate_content(
        model="gemini-2.5-flash-tts-preview",
        contents=prompt_tts,
        config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"))))
    )
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    return {"audio": audio_base64}