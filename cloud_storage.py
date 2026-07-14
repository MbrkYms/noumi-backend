import httpx, os
JSONBIN_URL = "https://api.jsonbin.io/v3/b/"
HEADERS = {"X-Master-Key": os.getenv("JSONBIN_KEY"), "Content-Type": "application/json"}

async def load_json(bin_id):
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(f"{JSONBIN_URL}{bin_id}/latest", headers=HEADERS)
        return r.json().get("record", {})

async def save_json(bin_id, data):
    async with httpx.AsyncClient(timeout=10.0) as c:
        await c.put(f"{JSONBIN_URL}{bin_id}", headers=HEADERS, json=data)