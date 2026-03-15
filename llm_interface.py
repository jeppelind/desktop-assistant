from collections import deque
from ollama import chat

class LLMInterface:
    def __init__(self):
        self.messages = deque(maxlen=50)
        self.model = "llama3.2"

    def query(self, input: str) -> str:
        self.messages.append({"role": "user", "content": input})
        response = chat(
            model=self.model, 
            messages=[*self.messages, {"role": "user", "content": input}], 
        )
        self.messages.append({"role": "assistant", "content": response.message.content})
        return response.message.content