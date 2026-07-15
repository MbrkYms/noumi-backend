import time
import psutil # <-- NOUVELLE DEP
import os
from datetime import datetime
from loguru import logger
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")

async def diagnose_self(latency_ms: float) -> dict:
    """Diagnostic complet de JARVIS pour le heartbeat"""
    
    # 1. RAM + CPU
    process = psutil.Process(os.getpid())
    ram_kb = process.memory_info().rss / 1024
    cpu_percent = process.cpu_percent(interval=0.1)
    
    # 2. TAILLE MONGODB
    db_size = 0
    if MONGO_URI:
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            db = client["stellia"]
            stats = db.command("dbstats")
            db_size = stats.get("dataSize", 0) / 1024 # en KB
        except: pass
    
    # 3. SCORE DE SANTÉ
    health_score = 10
    if latency_ms > 5000: health_score -= 3
    if ram_kb > 500000: health_score -= 2 # > 500MB
    if cpu_percent > 80: health_score -= 2
    
    etat = "OK" if health_score >= 8 else "DEGRADE" if health_score >= 5 else "CRITIQUE"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "latency_ms": int(latency_ms),
        "ram_kb": int(ram_kb),
        "cpu_percent": cpu_percent,
        "db_size_kb": int(db_size),
        "health_score": health_score,
        "etat": etat,
        "optimisations": []
    }
    
    # 4. PROPOSITIONS AUTO D'OPTIMISATION
    if latency_ms > 3000:
        report["optimisations"].append("Ajouter un cache Redis pour le RAG")
    if ram_kb > 400000:
        report["optimisations"].append("Limiter la taille de la mémoire RAG à 500 docs")
    if db_size > 100000: # > 100MB
        report["optimisations"].append("Archiver les anciens logs Mongo")
    
    logger.info(f"[DIAG] Etat: {etat} | Latency: {latency_ms}ms | RAM: {ram_kb/1024:.1f}MB")
    return report