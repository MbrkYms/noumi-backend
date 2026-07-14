import httpx, json, re, os

GEMINI_KEY = os.getenv("GEMINI_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY")

PROVIDERS = [
    {"name": "Gemini", "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"},
    {"name": "DeepSeek", "url": "https://api.deepseek.com/chat/completions", "key": DEEPSEEK_KEY, "model": "deepseek-chat"},
    {"name": "Groq", "url": "https://api.groq.com/openai/v1/chat/completions", "key": GROQ_KEY, "model": "llama-3.1-8b-instant"}
]

def _clean_json(text):
    # Si l'IA renvoie du texte avec ```json... ``` on nettoie
    text = text.strip().replace("```json", "").replace("```", "")
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    # Fallback: on renvoie le texte brut dans le champ text
    return {"text": text, "self": {}}

async def _call(p, prompt):
    headers = {"Content-Type": "application/json"}
    if p["name"]!= "Gemini":
        headers["Authorization"] = f"Bearer {p['key']}"

    # FORCER L'IA A REPONDRE EN JSON
    system_prompt = 'Tu es NOUMI, l\'IA personnelle de Yamine. Réponds TOUJOURS en JSON valide avec cette structure exacte: {"text": "ta réponse ici", "self": {}}'
    full_prompt = system_prompt + "\n\nQuestion: " + prompt

    payload = {"contents": [{"parts": [{"text": full_prompt}]}], "generationConfig": {"responseMimeType": "application/json"}} \
        if p["name"] == "Gemini" \
        else {"model": p["model"], "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}

    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.post(p["url"], headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"] if p["name"] == "Gemini" else data["choices"][0]["message"]["content"]
    return _clean_json(text)

async def ask_ai(prompt):
    for p in PROVIDERS:
        try:
            print(f"[ROUTER] {p['name']}")
            result = await _call(p, prompt)
            print(f"[ROUTER] OK: {result['text'][:50]}")
            return result
        except Exception as e:
            print(f"[ROUTER] {p['name']} KO: {e}")
    return {"text": "Mode Survie: Toutes les IA down", "self": {}}

# ===== ROUTE TTS GOOGLE =====
from fastapi import APIRouter, Request
router = APIRouter()

@router.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if not TTS_API_KEY:
        return {"error": "TTS_API_KEY manquante"}

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={TTS_API_KEY}"
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": "fr-FR", "name": "fr-FR-Wavenet-C"},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.1}
    }

    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.post(url, json=payload)
        r.raise_for_status()
        result = r.json()

    return {"audio": result['audioContent']}