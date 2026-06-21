import pytest

from tool_forge.schema import ToolCall, ToolSpec, VerificationOutcome
from tool_forge.verify import MalformedSpecError, VerificationResult, verify


@pytest.fixture
def registry() -> dict[str, ToolSpec]:
    WEATHER_SCHEMA = {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
        "additionalProperties": False,
    }

    return {
        "get_weather": ToolSpec(
            name="get_weather",
            description="Get weather based on location string",
            parameters=WEATHER_SCHEMA,
        )
    }

@pytest.mark.parametrize("bad_args", [{}, {"location": 123}, {"location": "Paris", "unit": "kelvin"}])
def test_schema_violations(registry: dict[str, ToolSpec], bad_args: dict[str, object]) -> None:
    """A known tool whose arguments violate its JSON Schema returns SCHEMA_VIOLATION."""
    result = verify(ToolCall(name="get_weather", arguments=bad_args), registry=registry)
    assert result.result is VerificationOutcome.SCHEMA_VIOLATION

def test_malformed_spec() -> None:
    """A tool whose own JSON Schema is invalid raises MalformedSpecError naming the tool, not a verdict."""
    bad_registry = {
        "get_weather": ToolSpec(
            name="get_weather",
            description="",
            parameters={"type": "Not a type"} # Invalid type for Schema
        )
    }
    with pytest.raises(MalformedSpecError) as exc_info:
        verify(ToolCall(name="get_weather", arguments={}), bad_registry)
    assert exc_info.value.name == "get_weather"

def test_valid_call(registry: dict[str, ToolSpec]) -> None:
    """A known tool with schema-conforming arguments returns VALID with no detail."""
    result = verify(ToolCall(name="get_weather", arguments={"location": "Paris"}), registry)
    assert result == VerificationResult(result=VerificationOutcome.VALID, detail=None)

def test_unknown_tool(registry: dict[str, ToolSpec]) -> None:
    """A call naming a tool absent from the registry returns UNKNOWN_TOOL."""
    result = verify(ToolCall(name="not_a_tool", arguments={}), registry)
    assert result.result is VerificationOutcome.UNKNOWN_TOOL

def test_verify_is_deterministic(registry: dict[str, ToolSpec]) -> None:
    """Verifying identical inputs twice yields equal results (the observable shadow of purity)."""
    call = ToolCall(name="get_weather", arguments={"location": "Paris"})
    assert verify(call, registry) == verify(call, registry)