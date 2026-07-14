import re, datetime
async def analyze_user(text, last_msg_time):
    stress = 0.5 + 0.1 * text.count("!") + 0.2 * len(re.findall(r'[A-Z]{3,}', text))
    return {"stress_level": min(stress, 1.0), "message_length": len(text.split()), "tone": "urgent" if "!" in text else "calme", "time_of_day": datetime.datetime.now().hour, "derailment_risk": 0.0}