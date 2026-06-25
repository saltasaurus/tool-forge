import pytest

from tool_forge.normalize import convert_type


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
