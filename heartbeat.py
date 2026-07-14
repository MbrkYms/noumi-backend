import asyncio, datetime
from auto_dev import run_auto_dev
async def heartbeat():
    while True:
        print(f"[{datetime.datetime.now()}] [HEARTBEAT] Cycle démarré")
        await run_auto_dev()
        await asyncio.sleep(600)