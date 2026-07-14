import httpx, json, re, os, base64
from google import genai
from google.genai import types # <-- AJOUTÉ

GEMINI_KEY = os.getenv("GEMINI_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY") # On peut le virer mais je le laisse

# CLIENT GEMINI NOUVEAU SDK
client = genai.Client(api_key=GEMINI_KEY)

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

async def _call_gemini(prompt, enable_search=False):
    system_instruction = 'Tu es STELLIA, l\'IA personnelle de Yamine. Si tu utilises internet, cite tes sources. Réponds TOUJOURS en JSON: {"text": "ta réponse", "self": {}, "sources": []}'
    full_prompt = system_instruction + "\n\nQuestion: " + prompt

    tools = []
    if enable_search:
        tools = [types.Tool(google_search=types.GoogleSearch())] # <-- FORMAT CORRECT

    response = client.models.generate_content(
        model="gemini-2.5-flash", # <-- TON MODÈLE STABLE
        contents=full_prompt,
        config=types.GenerateContentConfig(
            tools=tools,
            response_mime_type="application/json",
            temperature=0.9
        )
    )

    text = response.text
    sources = []
    # RECUP SOURCES GOOGLE SEARCH
    if response.candidates and response.candidates[0].grounding_metadata:
        for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
            if chunk.web:
                sources.append(chunk.web.uri)

    result = _clean_json(text)
    result["sources"] = sources
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
        print(f"[ROUTER] {p['name']} Status: {r.status_code}")
        if r.status_code!= 200: print(f"[ROUTER] ERREUR COMPLETE: {r.text}")
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        result = _clean_json(text)
        result["sources"] = []
        return result

async def ask_ai(prompt, enable_search=False):
    needs_search = any(k in prompt.lower() for k in ["actu", "météo", "prix", "cours", "aujourd'hui", "maintenant", "google", "recherche", "news"])

    if needs_search:
        try:
            print(f"[ROUTER] Tentative Gemini-2.5 SDK search=True")
            return await _call_gemini(prompt, enable_search=True)
        except Exception as e:
            print(f"[ROUTER] Gemini KO: {e}")
            last_error = str(e)
            return {"text": f"Bug Gemini: {last_error[:100]}", "self": {}, "sources": []}
    else:
        for p in PROVIDERS:
            try:
                print(f"[ROUTER] Tentative {p['name']} search=False")
                return await _call_rest(p, prompt)
            except Exception as e:
                print(f"[ROUTER] {p['name']} KO: {e}")
                last_error = str(e)
        return {"text": f"Bug: {last_error[:100]}", "self": {}, "sources": []}

# TTS GEMINI
from fastapi import APIRouter, Request
router = APIRouter()

@router.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text", "")

    # PROMPT PERSONA STELLIA AVEC TAGS
    prompt_tts = f"""# AUDIO PROFILE: Stellia
## THE SCENE: Appel vocal privé avec Yamine
### DIRECTOR'S NOTES
Style: Voix féminine française, chaleureuse, avec le sourire. Rythme naturel.
#### TRANSCRIPT
[text]"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-tts-preview", # <-- TTS GEMINI
        contents=prompt_tts.replace("[text]", text),
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore") # Voix chaleureuse
                )
            )
        )
    )

    # Récupération de l'audio
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    return {"audio": audio_base64}