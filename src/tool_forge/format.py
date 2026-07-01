from typing import Any

from tool_forge.schema import Conversation, ToolSpec

# HF / OpenAI function-calling shapes. Kept as loose dicts on purpose: this is the
# boundary type that apply_chat_template / SFTTrainer consume, not an internal type.
type ToolDef = dict[str, Any]          # {"type": "function", "function": {...}}
type Message = dict[str, Any]          # {"role": ..., "content"/"tool_calls": ...}
type FormattedExample = dict[str, Any]  # {"messages": [...], "tools": [...]}


def to_tools(tools: dict[str, ToolSpec]) -> list[ToolDef]:
    """Render the tool registry into function-spec dicts.

    Each ToolSpec -> {"type": "function", "function": {name, description, parameters}}.
    `parameters` is already JSON Schema (from normalize) -> pass it through untouched.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": toolspec.name,
                "description": toolspec.description,
                "parameters": toolspec.parameters
            }
        } for toolspec in tools.values()
    ]

def to_messages(conversation: Conversation) -> list[Message]:
    """Build the [user, assistant] message list for one single-turn example.

    user turn:      {"role": "user", "content": query}
    assistant turn: {"role": "assistant", "tool_calls": [...]}, one entry per gold call
                    ({"type": "function", "function": {"name", "arguments"}}).
    >1 gold call => parallel calls in a single assistant turn, in order.
    `arguments` stays a dict (the template JSON-encodes it; do not json.dumps).
    """
    
    user = {"role": "user", "content": conversation.query}

    calls = [{
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments,
        }} for call in conversation.gold_calls]

    assistant = {"role": "assistant", "tool_calls": calls}

    return [user, assistant]

def to_prompt_completion(row):
    """Separate a row of data into prompt, completion, and tools"""
    
    user, assistant = row["messages"]
    return {"prompt": [user], "completion": [assistant], "tools": row["tools"]}

def format_conversation(conversation: Conversation) -> FormattedExample:
    """Combine into the dict apply_chat_template / SFTTrainer consume."""
    
    return {
        "messages": to_messages(conversation=conversation),
        "tools": to_tools(conversation.tools)
        }
