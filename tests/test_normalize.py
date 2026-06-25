import json

import pytest

from tool_forge.normalize import (
    BASE_TYPES,
    XLAMParamSpec,
    convert_type,
    normalize_row,
    strip_modifiers,
    to_object_schema,
)
from tool_forge.schema import Conversation, ToolCall, ToolSpec


class TestConvertType:
    @pytest.mark.parametrize(
        "dtype,schematype",
        [("int", "integer"), ("float", "number"), ("bool", "boolean"), ("str", "string")],
    )
    def test_scalar_types(self, dtype: str, schematype: str) -> None:
        assert convert_type(dtype) == {"type": schematype}

    def test_bare_containers(self) -> None:
        assert convert_type("list") == {"type": "array"}
        assert convert_type("List") == {"type": "array"}
        assert convert_type("dict") == {"type": "object"}
        assert convert_type("set") == {"type": "array", "uniqueItems": True}

    def test_list_of_scalar(self) -> None:
        assert convert_type("List[int]") == {"type": "array", "items": {"type": "integer"}}

    def test_nested_list(self) -> None:
        assert convert_type("List[List[int]]") == {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        }

    def test_tuple(self) -> None:
        assert convert_type("Tuple[float, float]") == {
            "type": "array",
            "prefixItems": [{"type": "number"}, {"type": "number"}],
            "minItems": 2,
            "maxItems": 2,
        }

    def test_union(self) -> None:
        assert convert_type("Union[int, float]") == {"anyOf": [{"type": "integer"}, {"type": "number"}]}

    def test_list_of_union(self) -> None:
        assert convert_type("List[Union[int, float]]") == {
            "type": "array",
            "items": {"anyOf": [{"type": "integer"}, {"type": "number"}]},
        }

    def test_callable_is_permissive(self) -> None:
        assert convert_type("Callable[[float], float]") == {}

    def test_unknown_head_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            convert_type("FrozenSet[int]")

class TestStripModifiers:

    def test_no_modifier(self) -> None:
        res = strip_modifiers("str")

        assert res == ("str", False)

    def test_optional_modifier(self) -> None:
        res = strip_modifiers("str, optional")

        assert res == ("str", True)

    def test_optional_default_modifier(self) -> None:
        res = strip_modifiers("str, optional, default=100")

        assert res == ("str", True)

    def test_default_modifier(self) -> None:
        res = strip_modifiers("str, default='paris'")

        assert res == ("str", True)

    def test_no_container_modifier(self) -> None:
        res = strip_modifiers("Tuple[float, float]")

        assert res == ("Tuple[float, float]", False)

    def test_container_optional(self) -> None:
        res = strip_modifiers("List[Union[int, float]], optional")

        assert res == ("List[Union[int, float]]", True)


class TestToObjectSchema:
    def test_full_mapping(self) -> None:
        # One required, one optional-via-string, one optional-via-default-key.
        params: dict[str, XLAMParamSpec] = {
            "location": {"type": "str", "description": "City name"},
            "days": {"type": "int, optional", "description": "Forecast length"},
            "units": {"type": "str", "description": "Unit", "default": "celsius"},
        }

        result = to_object_schema(params)

        assert result == {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "description": "Forecast length"},
                "units": {"type": "string", "description": "Unit"},
            },
            "required": ["location"],
        }

    def test_container_param_carries_description(self) -> None:
        params: dict[str, XLAMParamSpec] = {"coords": {"type": "Tuple[float, float]", "description": "Lat/long"}}

        result = to_object_schema(params)

        assert result == {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "array",
                    "prefixItems": [{"type": "number"}, {"type": "number"}],
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Lat/long",
                },
            },
            "required": ["coords"],
        }

    def test_does_not_mutate_base_types(self) -> None:
        # Attaching `description` must not leak onto the shared BASE_TYPES fragment.
        params: dict[str, XLAMParamSpec] = {"days": {"type": "int", "description": "Forecast length"}}

        to_object_schema(params)

        assert BASE_TYPES["int"] == {"type": "integer"}
        assert "description" not in BASE_TYPES["int"]


class TestNormalizeRow:
    def test_real_xlam_row(self) -> None:
        # Mirrors ds["train"][0]: tools/answers are JSON-encoded strings; id/query are not.
        tools_obj = [
            {
                "name": "live_giveaways_by_type",
                "description": "Retrieve live giveaways from the GamerPower API based on the specified type.",
                "parameters": {
                    "type": {
                        "description": "The type of giveaways to retrieve (e.g., game, loot, beta).",
                        "type": "str",
                        "default": "game",
                    },
                },
            },
        ]
        answers_obj = [
            {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
            {"name": "live_giveaways_by_type", "arguments": {"type": "game"}},
        ]
        row = {
            "id": 0,
            "query": "Where can I find live giveaways for beta access and games?",
            "tools": json.dumps(tools_obj),
            "answers": json.dumps(answers_obj),
        }

        conv = normalize_row(row)

        assert conv == Conversation(
            id=0,
            query="Where can I find live giveaways for beta access and games?",
            tools={
                "live_giveaways_by_type": ToolSpec(
                    name="live_giveaways_by_type",
                    description="Retrieve live giveaways from the GamerPower API based on the specified type.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "The type of giveaways to retrieve (e.g., game, loot, beta).",
                            },
                        },
                        "required": [],  # the only param has a default -> optional
                    },
                ),
            },
            gold_calls=(
                ToolCall(name="live_giveaways_by_type", arguments={"type": "beta"}),
                ToolCall(name="live_giveaways_by_type", arguments={"type": "game"}),
            ),
        )