from enum import Enum

import jsonschema
from pydantic import BaseModel, ConfigDict

from .schema import ToolCall, ToolSpec


class VerificationOutcome(Enum):
    VALID = "VALID"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"

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