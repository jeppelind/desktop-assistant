from datetime import datetime

def get_current_time() -> str:
    """Get the current local time
    
    Returns:
        The current local time in HH:MM format
    """
    return datetime.now().strftime("%H:%M")
