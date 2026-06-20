from tools.local_time import get_current_time
from collections import deque
from ollama import chat


class LLMInterface:
    def __init__(self):
        self.messages = deque(maxlen=50)
        self.model = "gemma4:e2b"
        self.tools = [get_current_time]

    def query(self, input: str) -> str:
        self.messages.append({"role": "user", "content": input})
        response = chat(
            model=self.model, 
            messages=[*self.messages, {"role": "user", "content": input}],
            tools=self.tools
        )
        print(response)
        # self.messages.append({"role": "assistant", "content": response.message.content})
        self.messages.append(response.message)

        if response.message.tool_calls:
            tool_responses = self.generate_tool_response(response.message.tool_calls)
            self.messages.extend(tool_responses)
            final_response = chat(
                model=self.model, 
                messages=list(self.messages),
                tools=self.tools
            )
            print(final_response)
            # self.messages.append({"role": "assistant", "content": final_response.message.content})
            self.messages.append(final_response.message)
            print("TOOL CALL WAS USED")
            print(list(self.messages))
            return final_response.message.content
        
        print("Tool call not used")
        print(list(self.messages))
        return response.message.content
    
    def generate_tool_response(self, tool_calls) -> deque:
        result_list = deque()
        for call in tool_calls:
            if call.function.name == "get_current_time":
                result = get_current_time()
            else:
                result = "Tool not found"
            result_list.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})
        return result_list