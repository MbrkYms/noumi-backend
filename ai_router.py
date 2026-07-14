import httpx, json, re, os
GEMINI_KEY = os.getenv("GEMINI_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY")

PROVIDERS = [
    # 1. GROQ EN PREMIER - Le plus rapide mais PAS de recherche
    {"name": "Groq", "url": "https://api.groq.com/openai/v1/chat/completions", "key": GROQ_KEY, "model": "llama-3.1-8b-instant", "search": False},
    # 2. GEMINI EN 2EME - Lui peut chercher
    {"name": "Gemini", "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}", "search": True},
    # 3. DEEPSEEK EN DERNIER - PAS de recherche
    {"name": "DeepSeek", "url": "https://api.deepseek.com/chat/completions", "key": DEEPSEEK_KEY, "model": "deepseek-chat", "search": False}
]

def _clean_json(text):
    text = text.strip().replace("```json", "").replace("```", "")
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return {"text": text, "self": {}}

async def _call(p, prompt, enable_search=False):
    headers = {"Content-Type": "application/json"}
    if p["name"]!= "Gemini":
        headers["Authorization"] = f"Bearer {p['key']}"

    system_prompt = 'Tu es STELLIA, l\'IA personnelle de Yamine. Tu te souviens que Yamine s\'appelle Yamine. Si on te demande une info actuelle, actu, météo, prix, tu dois le dire. Réponds TOUJOURS en JSON: {"text": "ta réponse", "self": {}, "sources": []}'

    # ===== FIX GEMINI AVEC RECHERCHE =====
    if p["name"] == "Gemini":
        parts = [{"text": system_prompt + "\n\nQuestion: " + prompt}]
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        # ON AJOUTE L'OUTIL DE RECHERCHE SI DEMANDÉ
        if enable_search and p.get("search"):
            payload["tools"] = [{"google_search": {}}]
    else:
        payload = {
            "model": p["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.post(p["url"], headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

        if p["name"] == "Gemini":
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            # ON RÉCUPÈRE LES SOURCES
            sources = []
            if "groundingMetadata" in data["candidates"][0]:
                for s in data["candidates"][0]["groundingMetadata"].get("groundingChunks", []):
                    if "web" in s: sources.append(s["web"]["title"])
            result = _clean_json(text)
            result["sources"] = sources
            return result
        else:
            text = data["choices"][0]["message"]["content"]
            return _clean_json(text)

async def ask_ai(prompt, enable_search=False): # AJOUT PARAM
    last_error = ""
    for p in PROVIDERS:
        try:
            print(f"[ROUTER] {p['name']} Search={enable_search}")
            # SI C'EST UNE QUESTION QUI NÉCESSITE LA RECHERCHE, ON FORCE GEMINI
            needs_search = any(k in prompt.lower() for k in ["actu", "météo", "prix", "cours", "aujourd'hui", "maintenant", "google", "recherche"])
            use_search = enable_search or needs_search

            if use_search and p["name"]!= "Gemini":
                continue # On skip Groq/Deepseek si on a besoin de search

            result = await _call(p, prompt, enable_search=use_search)
            return result
        except Exception as e:
            print(f"[ROUTER] {p['name']} KO: {e}")
            last_error = str(e)
    return {"text": f"Erreur IA: {last_error[:50]}. Réessaie.", "self": {}, "sources": []}

# TTS reste identique
from fastapi import APIRouter, Request
router = APIRouter()

@router.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if not TTS_API_KEY: return {"error": "TTS_API_KEY manquante"}
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={TTS_API_KEY}"
    payload = {"input": {"text": text}, "voice": {"languageCode": "fr-FR", "name": "fr-FR-Wavenet-C"}, "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.05}}
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.post(url, json=payload); r.raise_for_status(); result = r.json()
        return {"audio": result['audioContent']}