from ollama import Message
from tools.local_time import get_current_time
from collections import deque
from ollama import chat

class LLMInterface:
    def __init__(self):
        self.messages = deque(maxlen=50)
        self.model = "gemma4:e2b"
        self.tool_functions = {
          'get_current_time': get_current_time
        }
        self.tools = list(self.tool_functions.values())

    def query(self, input: str) -> str:
        self.messages.append({"role": "user", "content": input})
        response = self.generate_response()
        return response.content

    def generate_response(self) -> Message:
        response = chat(
            model=self.model, 
            messages=list(self.messages),
            tools=self.tools,
            think=True
        )
        self.messages.append(response.message)

        if response.message.tool_calls:
            tool_responses = self.generate_tool_response(response.message.tool_calls)
            self.messages.extend(tool_responses)
            return self.generate_response()
        print(list(self.messages))
        return response.message
    
    def generate_tool_response(self, tool_calls) -> deque:
        result_list = deque()
        for call in tool_calls:
            if call.function.name in self.tool_functions:
                result = self.tool_functions[call.function.name](**call.function.arguments)
            else:
                result = "Tool not found"
            result_list.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})
        return result_list
