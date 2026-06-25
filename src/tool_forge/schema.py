from enum import Enum

from pydantic import BaseModel, ConfigDict


class VerificationOutcome(Enum):
    VALID = "VALID"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"

class ToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    parameters: dict[str, object] # JSON Schema Object

class ToolCall(BaseModel):

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    arguments: dict[str, object]
    id: str | None = None

class PreferencePair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    tools: dict[str, ToolSpec]
    chosen: ToolCall
    rejected: ToolCall
    rejection_reason: VerificationOutcome

class Conversation(BaseModel):
    """One single-turn SFT example: a user query, the tools in scope, and the gold call(s)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    query: str
    tools: dict[str, ToolSpec]          # registry keyed by name -> matches verify()'s signature
    gold_calls: tuple[ToolCall, ...]    # 1 = single call, >1 = parallel calls