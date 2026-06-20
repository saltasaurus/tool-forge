from pydantic import BaseModel, ConfigDict


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

