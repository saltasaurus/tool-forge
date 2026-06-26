import pytest

from tool_forge.schema import Conversation, ToolCall, ToolSpec
from tool_forge.verify import keep_valid

WEATHER = ToolSpec(
    name="get_weather",
    description="Get weather",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
)
REGISTRY = {"get_weather": WEATHER}


def _convo(cid: int, *calls: ToolCall) -> Conversation:
    return Conversation(id=cid, query="q", tools=REGISTRY, gold_calls=calls)


VALID_CALL = ToolCall(name="get_weather", arguments={"location": "Paris"})
SCHEMA_VIOLATION = ToolCall(name="get_weather", arguments={"location": 123})  # int, not str
UNKNOWN_TOOL = ToolCall(name="nonesuch", arguments={})


def test_empty_input() -> None:
    assert keep_valid([]) == ([], 0)


def test_all_valid_conversation_kept() -> None:
    convos = [_convo(1, VALID_CALL)]
    survivors, quarantined = keep_valid(convos)
    assert survivors == convos
    assert quarantined == 0


def test_schema_violation_quarantined() -> None:
    survivors, quarantined = keep_valid([_convo(1, SCHEMA_VIOLATION)])
    assert survivors == []
    assert quarantined == 1


def test_unknown_tool_quarantined() -> None:
    survivors, quarantined = keep_valid([_convo(1, UNKNOWN_TOOL)])
    assert survivors == []
    assert quarantined == 1


def test_any_bad_call_sinks_the_whole_conversation() -> None:
    """Parallel calls: one valid + one bad => the conversation is quarantined, not split."""
    survivors, quarantined = keep_valid([_convo(1, VALID_CALL, SCHEMA_VIOLATION)])
    assert survivors == []
    assert quarantined == 1


def test_mixed_list_preserves_survivor_order_and_counts() -> None:
    good_a = _convo(1, VALID_CALL)
    bad = _convo(2, SCHEMA_VIOLATION)
    good_b = _convo(3, VALID_CALL, VALID_CALL)
    survivors, quarantined = keep_valid([good_a, bad, good_b])
    assert survivors == [good_a, good_b]  # order preserved, bad dropped
    assert quarantined == 1
