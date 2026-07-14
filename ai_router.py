import httpx, json, re, os
GEMINI_KEY = os.getenv("GEMINI_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY")

PROVIDERS = [
    {"name": "Groq", "url": "https://api.groq.com/openai/v1/chat/completions", "key": GROQ_KEY, "model": "llama-3.1-8b-instant", "search": False},
    {"name": "Gemini", "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}", "search": True},
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

    system_prompt = 'Tu es STELLIA, l\'IA personnelle de Yamine. Tu te souviens que Yamine s\'appelle Yamine. Si on te demande une info actuelle, actu, météo, prix, tu dois chercher. Réponds TOUJOURS en JSON: {"text": "ta réponse", "self": {}, "sources": []}'

    # ===== FIX GEMINI AVEC RECHERCHE =====
    if p["name"] == "Gemini":
        parts = [{"text": system_prompt + "\n\nQuestion: " + prompt}]
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        # FORMAT OFFICIEL GOOGLE POUR SEARCH
        if enable_search and p.get("search"):
            payload["tools"] = [{"google_search_retrieval": {}}] # <-- C'ETAIT ÇA LE BUG

    else:
        payload = {
            "model": p["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

    print(f"[ROUTER] Payload envoyé à {p['name']}: search={enable_search}") # LOG

    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.post(p["url"], headers=headers, json=payload)
        if r.status_code!= 200:
            print(f"[ROUTER] ERREUR {p['name']}: {r.text}") # LOG ERREUR COMPLETE
        r.raise_for_status()
        data = r.json()

        if p["name"] == "Gemini":
            text = data["candidates"][0]["content"]["parts"][0]["text"]
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

async def ask_ai(prompt, enable_search=False):
    last_error = ""
    for p in PROVIDERS:
        try:
            needs_search = any(k in prompt.lower() for k in ["actu", "météo", "prix", "cours", "aujourd'hui", "maintenant", "google", "recherche", "temps"])

            # SI BESOIN DE RECHERCHE ON FORCE GEMINI DIRECT
            if needs_search and p["name"]!= "Gemini":
                continue

            use_search = enable_search or needs_search
            print(f"[ROUTER] Tentative {p['name']} search={use_search}")
            result = await _call(p, prompt, enable_search=use_search)
            return result
        except Exception as e:
            print(f"[ROUTER] {p['name']} KO: {e}")
            last_error = str(e)

    # FALLBACK SI TOUT PLANTE
    return {"text": f"Désolée Yamine, j'ai eu un bug réseau. Réessaie.", "self": {}, "sources": []}

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