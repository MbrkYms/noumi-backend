import re
from datetime import datetime # FIX: import direct

async def analyze_user(text: str, last_msg_time: datetime) -> dict:
    """Analyse l'état émotionnel de Monsieur pour adapter JARVIS"""
    
    text_lower = text.lower()
    
    # 1. STRESS
    stress = 0.5
    stress += 0.1 * text.count("!") 
    stress += 0.2 * len(re.findall(r'[A-Z]{3,}', text)) # MAJUSCULES
    stress += 0.3 if any(k in text_lower for k in ["urgent", "vite", "bug", "erreur", "crash"]) else 0
    
    # 2. FATIGUE - Si message tard le soir
    hour = datetime.now().hour
    fatigue = 0.8 if hour >= 23 or hour <= 6 else 0.2
    
    # 3. CURIOSITÉ
    curiosity = 0.5
    curiosity += 0.2 * text.count("?")
    curiosity += 0.3 if any(k in text_lower for k in ["pourquoi", "comment", "explique"]) else 0
    
    # 4. TONE
    if "!" in text: tone = "urgent"
    elif "?" in text: tone = "curieux"
    elif fatigue > 0.7: tone = "fatigué"
    else: tone = "calme"
    
    # 5. RISQUE DE DÉRIVE
    derailment_risk = 0.0
    if stress > 0.8 and "patch" in text_lower: derailment_risk = 0.9 # Demande patch sous stress
    
    return {
        "stress_level": min(stress, 1.0),
        "fatigue_level": fatigue,
        "curiosity_level": min(curiosity, 1.0),
        "message_length": len(text.split()),
        "tone": tone,
        "time_of_day": hour,
        "derailment_risk": derailment_risk
    }