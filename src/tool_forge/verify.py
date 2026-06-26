import jsonschema
from pydantic import BaseModel, ConfigDict

from .schema import Conversation, ToolCall, ToolSpec, VerificationOutcome


class VerificationResult(BaseModel):

    model_config = ConfigDict(frozen=True, extra="forbid")

    result: VerificationOutcome
    detail: str | None

class MalformedSpecError(Exception):
    def __init__(self, name: str, error: str) -> None:
        self.name = name
        self.error = error
        super().__init__(f"{self.name} | {self.error}")

def verify(call: ToolCall, registry: dict[str, ToolSpec]) -> VerificationResult:
    if call.name not in registry:
        return VerificationResult(
            result=VerificationOutcome.UNKNOWN_TOOL,
            detail=f"{call.name} not found in registry [{list(registry.keys())}]")
    spec = registry[call.name]
    try:
        jsonschema.validate(call.arguments, spec.parameters)
    except jsonschema.ValidationError as validation_err:
        return VerificationResult(
            result=VerificationOutcome.SCHEMA_VIOLATION,
            detail=str(validation_err))
    except jsonschema.SchemaError as schema_error:
        # Crash to prevent model training on invalid inputs. 
        # Catch before training starts
        raise MalformedSpecError(call.name, str(schema_error)) from schema_error

    return VerificationResult(
            result=VerificationOutcome.VALID,
            detail=None
        )

def keep_valid(conversations: list[Conversation]) -> tuple[list[Conversation], int]:
    """Partition conversations into trainable vs quarantined (Phase-1 use of verify).

    A conversation survives iff EVERY gold call verifies VALID against its own
    `tools` registry. Returns (survivors, quarantined_count); survivor order is
    preserved. Pure — no IO.

    `MalformedSpecError` (a tool's own JSON Schema is invalid) propagates by design:
    that is a pipeline bug to fix before training, not row-level noise to drop.
    """
    survivors: list[Conversation] = []
    quarantined = 0
    for convo in conversations:
        if all(
            verify(call, convo.tools).result is VerificationOutcome.VALID
            for call in convo.gold_calls
        ):
            survivors.append(convo)
        else:
            quarantined += 1
    return survivors, quarantined