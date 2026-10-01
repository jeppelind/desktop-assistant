from tools import local_time

TOOLS = [
    (local_time.get_current_time, local_time.SCHEMA),
]

tool_functions = {schema["function"]["name"]: fn for fn, schema in TOOLS}
tool_schemas = [schema for _, schema in TOOLS]
