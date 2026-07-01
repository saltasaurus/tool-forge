from typing import Any

import pytest

from tool_forge.format import format_conversation, to_messages, to_tools
from tool_forge.schema import Conversation, ToolCall, ToolSpec

WEATHER_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"location": {"type": "string"}},
    "required": ["location"],
}
TIME_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"tz": {"type": "string"}},
    "required": ["tz"],
}


@pytest.fixture
def registry() -> dict[str, ToolSpec]:
    return {
        "get_weather": ToolSpec(
            name="get_weather", description="Get weather", parameters=WEATHER_PARAMS
        ),
        "get_time": ToolSpec(
            name="get_time", description="Get time", parameters=TIME_PARAMS
        ),
    }


@pytest.fixture
def convo(registry: dict[str, ToolSpec]) -> Conversation:
    return Conversation(
        id=1,
        query="weather in Paris?",
        tools=registry,
        gold_calls=(ToolCall(name="get_weather", arguments={"location": "Paris"}),),
    )


# --- to_tools -------------------------------------------------------------

def test_to_tools_one_def_per_spec(registry: dict[str, ToolSpec]) -> None:
    tools = to_tools(registry)
    assert len(tools) == len(registry)
    assert all(t["type"] == "function" for t in tools)


def test_to_tools_carries_name_desc_and_schema(registry: dict[str, ToolSpec]) -> None:
    by_name = {t["function"]["name"]: t["function"] for t in to_tools(registry)}
    assert by_name["get_weather"]["description"] == "Get weather"
    # parameters passed through as the SAME JSON Schema object, not stringified.
    assert by_name["get_weather"]["parameters"] == WEATHER_PARAMS


# --- to_messages ----------------------------------------------------------

def test_to_messages_is_user_then_assistant(convo: Conversation) -> None:
    messages = to_messages(convo)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "weather in Paris?"


def test_to_messages_single_call_arguments_stay_dict(convo: Conversation) -> None:
    assistant = to_messages(convo)[1]
    calls = assistant["tool_calls"]
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "get_weather"
    assert fn["arguments"] == {"location": "Paris"}  # dict, not a JSON string
    assert isinstance(fn["arguments"], dict)


def test_to_messages_parallel_calls_in_one_turn(registry: dict[str, ToolSpec]) -> None:
    convo = Conversation(
        id=2,
        query="weather and time in Paris?",
        tools=registry,
        gold_calls=(
            ToolCall(name="get_weather", arguments={"location": "Paris"}),
            ToolCall(name="get_time", arguments={"tz": "Europe/Paris"}),
        ),
    )
    messages = to_messages(convo)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    names = [c["function"]["name"] for c in messages[1]["tool_calls"]]
    assert names == ["get_weather", "get_time"]  # order preserved


# --- format_conversation --------------------------------------------------

def test_format_conversation_combines_parts(convo: Conversation) -> None:
    out = format_conversation(convo)
    assert out["messages"] == to_messages(convo)
    assert out["tools"] == to_tools(convo.tools)
