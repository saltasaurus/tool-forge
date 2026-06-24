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
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    query: str
    tools: dict[str, ToolSpec]
    gold_calls: tuple[ToolCall, ...]