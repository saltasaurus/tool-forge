import json
from pathlib import Path

from tool_forge.eval import RowScore, parse_calls, score_completions, score_row
from tool_forge.schema import ToolCall, ToolSpec

REGISTRY = {
    "get_weather": ToolSpec(
        name="get_weather",
        description="Get weather",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )
}
GOLD = [ToolCall(name="get_weather", arguments={"location": "Paris"})]
_WRAPPED = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
_BARE = '{"name": "get_weather", "arguments": {"location": "Paris"}}'


def _score(pred: list[ToolCall], *, emits_json: bool = True, wrapped: bool = True) -> RowScore:
    return score_row(pred, GOLD, REGISTRY, emits_json=emits_json, wrapped=wrapped)


# --- parse_calls ----------------------------------------------------------


def test_parses_wrapped_call() -> None:
    calls, ok, wrapped = parse_calls(_WRAPPED)
    assert ok and wrapped and calls == GOLD


def test_parses_bare_json_fallback() -> None:
    # base models emit the call as bare JSON, no <tool_call> tags
    calls, ok, wrapped = parse_calls(_BARE)
    assert ok and not wrapped and calls == GOLD


def test_bare_json_stops_at_garbage() -> None:
    # correct call, then the no-EOS ramble the base model degenerates into
    calls, ok, wrapped = parse_calls(_BARE + "\n fø\n føuser\nWhat is ...")
    assert ok and not wrapped and calls == GOLD


def test_no_call_at_all() -> None:
    calls, ok, wrapped = parse_calls("I cannot help with that.")
    assert calls == [] and not ok and not wrapped


def test_malformed_wrapped_flips_flag() -> None:
    calls, ok, wrapped = parse_calls("<tool_call>{not json}</tool_call>")
    assert calls == [] and not ok and wrapped


# --- score_row ladder -----------------------------------------------------


def test_wrapped_correct_is_strict() -> None:
    s = _score(GOLD, wrapped=True)
    assert s.name_and_args and s.tool_name and s.schema_valid
    assert s.protocol and s.strict and not s.hallucinated


def test_bare_correct_is_content_not_strict() -> None:
    # right content, no wrapper -> reasoning rungs pass, protocol/strict fail
    s = _score(GOLD, wrapped=False)
    assert s.name_and_args and s.tool_name and s.schema_valid
    assert not s.protocol and not s.strict


def test_right_name_wrong_args() -> None:
    s = _score([ToolCall(name="get_weather", arguments={"location": "London"})])
    assert s.tool_name and not s.name_and_args and s.schema_valid and not s.strict


def test_hallucinated_tool() -> None:
    s = _score([ToolCall(name="teleport", arguments={"location": "Paris"})])
    assert s.hallucinated and not s.schema_valid and not s.tool_name


def test_schema_violation_is_not_valid() -> None:
    # right tool, but arguments violate the schema (location must be a string)
    s = _score([ToolCall(name="get_weather", arguments={"location": 42})])
    assert not s.schema_valid and not s.hallucinated


# --- score_completions (vLLM dump path) -----------------------------------

_CALL = {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Paris"}}}
_TOOL = {"type": "function", "function": REGISTRY["get_weather"].model_dump()}
_DEV_ROW = {
    "messages": [
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "tool_calls": [_CALL]},
    ],
    "tools": [_TOOL],
}


def test_score_completions_bare_json_scores_content(tmp_path: Path) -> None:
    data = tmp_path / "dev.jsonl"
    dump = tmp_path / "gen.jsonl"
    data.write_text(json.dumps(_DEV_ROW) + "\n")
    # base-style bare JSON dump: content correct, protocol absent
    dump.write_text(json.dumps({"i": 0, "completion": _BARE}) + "\n")

    scores, metrics = score_completions(data, dump)
    assert len(scores) == 1
    assert metrics["name_and_args"] == 1.0
    assert metrics["strict"] == 0.0
