import httpx, json, re, os
GEMINI_KEY = os.getenv("GEMINI_KEY"); DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY"); GROQ_KEY = os.getenv("GROQ_KEY")
PROVIDERS = [
    {"name": "Gemini", "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"},
    {"name": "DeepSeek", "url": "https://api.deepseek.com/chat/completions", "key": DEEPSEEK_KEY, "model": "deepseek-chat"},
    {"name": "Groq", "url": "https://api.groq.com/openai/v1/chat/completions", "key": GROQ_KEY, "model": "llama-3.1-8b-instant"}
]
def _clean_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL); return json.loads(match.group(0)) if match else {"text": text, "error": "No JSON"}
async def _call(p, prompt):
    headers = {"Content-Type": "application/json"};
    if p["name"]!= "Gemini": headers["Authorization"] = f"Bearer {p['key']}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}} if p["name"] == "Gemini" else {"model": p["model"], "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
    async with httpx.AsyncClient(timeout=20.0) as c: r = await c.post(p["url"], headers=headers, json=payload); r.raise_for_status(); data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"] if p["name"] == "Gemini" else data["choices"][0]["message"]["content"]; return _clean_json(text)
async def ask_ai(prompt):
    for p in PROVIDERS:
        try: print(f"[ROUTER] {p['name']}"); return await _call(p, prompt)
        except Exception as e: print(f"[ROUTER] {p['name']} KO: {e}")
    return {"text": "Mode Survie: Toutes les IA down", "self": {}}