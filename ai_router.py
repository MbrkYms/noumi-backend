import httpx, json, re, os, base64
from google import genai
from google.genai import types

# ON CHARGE TOUTES LES CLÉS GEMINI
GEMINI_KEYS = [k for k in [
    os.getenv("GEMINI_KEY"),
    os.getenv("GEMINI_KEY_2"),
    os.getenv("GEMINI_KEY_3")
] if k]

DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")

PROVIDERS = [
    {"name": "Groq", "url": "https://api.groq.com/openai/v1/chat/completions", "key": GROQ_KEY, "model": "llama-3.1-8b-instant", "search": False},
    {"name": "DeepSeek", "url": "https://api.deepseek.com/chat/completions", "key": DEEPSEEK_KEY, "model": "deepseek-chat", "search": False}
]

def _clean_json(text):
    text = text.strip().replace("```json", "").replace("```", "")
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return {"text": text, "self": {}}

async def _call_gemini_with_key(api_key, key_index, prompt, enable_search=False):
    client = genai.Client(api_key=api_key)

    system_instruction = 'Tu es STELLIA, l\'IA personnelle de Yamine. Si tu utilises internet, cite tes sources. Réponds TOUJOURS en JSON: {"text": "ta réponse", "self": {}, "sources": []}'
    full_prompt = system_instruction + "\n\nQuestion: " + prompt

    tools = []
    if enable_search:
        tools = [types.Tool(google_search=types.GoogleSearch())]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            tools=tools,
            response_mime_type="application/json",
            temperature=0.9
        )
    )

    text = response.text
    sources = []
    if response.candidates and response.candidates[0].grounding_metadata:
        for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
            if chunk.web: sources.append(chunk.web.uri)

    result = _clean_json(text)
    result["sources"] = sources
    result["model_used"] = f"Gemini-2.5-Flash [Clé {key_index}]" # <-- AJOUT
    return result

async def _call_rest(p, prompt):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {p['key']}"}
    system_prompt = 'Tu es STELLIA, l\'IA personnelle de Yamine. Réponds TOUJOURS en JSON: {"text": "ta réponse", "self": {}, "sources": []}'
    payload = {
        "model": p["model"],
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    async with httpx.AsyncClient(timeout=25.0) as c:
        r = await c.post(p["url"], headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        result = _clean_json(text)
        result["sources"] = []
        result["model_used"] = p["name"] # <-- AJOUT
        return result

async def ask_ai(prompt, enable_search=False):
    needs_search = any(k in prompt.lower() for k in ["actu", "météo", "prix", "cours", "aujourd'hui", "maintenant", "google", "recherche", "news"])

    if needs_search:
        last_error = ""
        # ON TESTE TOUTES LES CLÉS GEMINI UNE PAR UNE
        for i, key in enumerate(GEMINI_KEYS):
            try:
                print(f"[ROUTER] Tentative Gemini Clé {i+1}/{len(GEMINI_KEYS)} search=True")
                return await _call_gemini_with_key(key, i+1, prompt, enable_search=True)
            except Exception as e:
                error_str = str(e)
                print(f"[ROUTER] Gemini Clé {i+1} KO: {error_str[:50]}")
                last_error = error_str
                if "429" not in error_str: # Si c'est pas un quota, on stop
                    break

        # SI TOUTES LES CLÉS SONT HS -> FALLBACK
        print("[ROUTER] Toutes les clés Gemini HS. Fallback Groq")
        fallback_text = "Désolé Yamine [sighs] J'ai plus de quota Google sur toutes mes clés. Je réponds sans recherche pour l'instant."
        return {"text": fallback_text, "self": {}, "sources": [], "model_used": "Aucun - Quota HS"} # <-- AJOUT

    else: # PAS BESOIN DE RECHERCHE
        for p in PROVIDERS:
            try:
                print(f"[ROUTER] Tentative {p['name']} search=False")
                return await _call_rest(p, prompt)
            except Exception as e:
                print(f"[ROUTER] {p['name']} KO: {e}")
        return {"text": "Toutes les IA sont down", "self": {}, "sources": [], "model_used": "Aucun"} # <-- AJOUT

# TTS GEMINI - utilise la première clé dispo
from fastapi import APIRouter, Request
router = APIRouter()

@router.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if not GEMINI_KEYS: return {"error": "Aucune GEMINI_KEY trouvée"}

    client = genai.Client(api_key=GEMINI_KEYS[0])

    prompt_tts = f"""# AUDIO PROFILE: Stellia
## THE SCENE: Appel vocal privé avec Yamine
### DIRECTOR'S NOTES
Style: Voix féminine française, chaleureuse, avec le sourire
#### TRANSCRIPT
{text}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-tts-preview",
        contents=prompt_tts,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            )
        )
    )

    audio_data = response.candidates[0].content.parts[0].inline_data.data
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    return {"audio": audio_base64}