from tool_forge.schema import Conversation
from tool_forge.split import split


def _make(n: int) -> list[Conversation]:
    """n throwaway Conversations with distinct ids; only the id matters here."""
    return [
        Conversation(id=i, query=f"q{i}", tools={}, gold_calls=())
        for i in range(n)
    ]


def test_partition_is_complete_and_disjoint() -> None:
    """Every example lands in exactly one split — nothing lost, nothing leaked."""
    convos = _make(1000)
    train, val, test = split(convos)

    assert len(train) + len(val) + len(test) == len(convos)
    ids = [{c.id for c in part} for part in (train, val, test)]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()
    assert ids[0] | ids[1] | ids[2] == {c.id for c in convos}


def test_proportions_are_roughly_80_10_10() -> None:
    train, val, test = split(_make(1000))
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100


def test_deterministic_for_a_fixed_seed() -> None:
    """Same seed -> same partition, so the artifact is reproducible."""
    a = split(_make(1000), seed=42)
    b = split(_make(1000), seed=42)
    assert [{c.id for c in p} for p in a] == [{c.id for c in p} for p in b]
