from datetime import datetime

SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current local time in HH:MM format",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

def get_current_time() -> str:
    """Get the current local time
    
    Returns:
        The current local time in HH:MM format
    """
    return datetime.now().strftime("%H:%M")
