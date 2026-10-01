import json
from openai import OpenAI, OpenAIError
from tools.registry import tool_functions, tool_schemas

MAX_TOOL_ITERATIONS = 5
MAX_TURNS = 12

class LLMInterface:
    def __init__(self):
        self.messages: list = []
        self.client = OpenAI(base_url="http://127.0.0.1:1337/v1", api_key="jan", timeout=60)
        self.model = "Jan-v3.5-4B-Q4_K_XL"
        self.tool_functions = tool_functions
        self.tools = tool_schemas

    def query(self, user_input: str) -> str:
        self._trim_history()
        start = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            return self._generate_response()
        except OpenAIError as e:
            print(f"LLM request failed: {e}")
            del self.messages[start:]
            return "Sorry, I couldn't reach the language model."

    def _trim_history(self):
        turns_to_keep = MAX_TURNS - 1
        user_indices = [idx for idx, msg in enumerate(self.messages) if msg["role"] == "user"]
        if len(user_indices) >= turns_to_keep:
            self.messages = self.messages[user_indices[-turns_to_keep]:]

    def _generate_response(self) -> str:
        for i in range(MAX_TOOL_ITERATIONS):
            kwargs = {}
            if i == MAX_TOOL_ITERATIONS - 1:
                kwargs["tool_choice"] = "none"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                **kwargs
            )
            message = response.choices[0].message
            entry = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            self.messages.append(entry)

            if not message.tool_calls:
                return entry["content"]
            self.messages.extend(self._generate_tool_response(message.tool_calls))
        return "I couldn't complete that request."

    def _generate_tool_response(self, tool_calls) -> list:
        result_list = []
        for call in tool_calls:
            name = call.function.name
            if name in self.tool_functions:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    result = self.tool_functions[name](**arguments)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = "Tool not found"
            result_list.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
        return result_list
