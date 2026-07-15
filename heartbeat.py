import asyncio
from datetime import datetime # FIX
from loguru import logger
from ai_router import heartbeat as jarvis_heartbeat # On importe le VRAI heartbeat

async def heartbeat_logger(): # Renommé pour pas faire conflit
    while True:
        logger.info(f"[HEARTBEAT-LOGGER] Cycle démarré à {datetime.now()}")
        await jarvis_heartbeat() # On appelle le vrai
        await asyncio.sleep(1800) # 30min