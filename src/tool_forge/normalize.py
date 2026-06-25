import logging

logger = logging.getLogger(__name__)

type JSONSchema = dict[str, object]

# Leaf table: bare (bracketless) type strings -> JSON Schema fragment.
# A type lives here only if it can appear *without* brackets in the corpus
# (str/int/float/bool, and bare list/List/dict/Dict/set).
BASE_TYPES: dict[str, JSONSchema] = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "set": {"type": "array", "uniqueItems": True},
    "list": {"type": "array"},
    "List": {"type": "array"},
    "dict": {"type": "object"},
    "Dict": {"type": "object"},
}


def _split_top_level(inner: str) -> list[str]:
    """Split ``inner`` on commas at bracket depth 0 so nested generics stay intact.

    ``"int, float"`` -> ``["int", "float"]`` but ``"Dict[str, int], str"`` ->
    ``["Dict[str, int]", "str"]`` (the comma inside the brackets is not a separator).
    """
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[start:i])
            start = i + 1
    parts.append(inner[start:])
    return [p.strip() for p in parts]


def convert_type(type_str: str) -> JSONSchema:
    """Convert an xLAM parameter type string into a JSON Schema fragment.

    Recursive: bracketless types are leaves (table lookup); ``Head[...]`` types
    dispatch on the head and recurse on the inner type(s).
    """
    type_str = type_str.strip()

    # Base case: no brackets -> a leaf type.
    if "[" not in type_str:
        return BASE_TYPES[type_str]

    # Recursive case: split "Head[Inner]" using the *last* ']' to keep nesting balanced.
    head = type_str[: type_str.index("[")].strip()
    inner = type_str[type_str.index("[") + 1 : type_str.rindex("]")]

    match head.lower():
        case "list":
            return {"type": "array", "items": convert_type(inner)}
        case "tuple":
            # Fixed-length, positionally-typed array -> prefixItems + length pin.
            members = [convert_type(p) for p in _split_top_level(inner)]
            return {
                "type": "array",
                "prefixItems": members,
                "minItems": len(members),
                "maxItems": len(members),
            }
        case "union":
            return {"anyOf": [convert_type(p) for p in _split_top_level(inner)]}
        case "callable":
            # A function has no JSON representation. Emit a
            # permissive schema (accepts anything) and log it for coverage tracking.
            logger.warning("Non-representable type %r -> permissive schema", type_str)
            return {}
        case _:
            raise NotImplementedError(head)

def strip_modifiers(type_str: str) -> tuple[str, bool]:
    """Return (clean type expression, is_optional)."""

    parts = _split_top_level(type_str)
    type_expr = parts[0]
    modifiers = parts[1:]
    is_optional = any(m == "optional" or m.startswith("default") for m in modifiers)
    return type_expr, is_optional