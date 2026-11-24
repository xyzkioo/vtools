"""Normalize positional and keyword arguments provided by a vision adapter."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InputBundle:
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    description: str = "custom"

    @classmethod
    def from_value(cls, value: Any, batch_size: int = 1, description: str = "custom") -> "InputBundle":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict) and "args" in value:
            raw_args = value["args"]
            # An adapter can pass a list of detection images as one argument.
            return cls(
                args=raw_args if isinstance(raw_args, tuple) else (raw_args,),
                kwargs=dict(value.get("kwargs", {})),
                batch_size=int(value.get("batch_size", batch_size)),
                description=str(value.get("description", description)),
            )
        args = value if isinstance(value, tuple) else (value,)
        return cls(args, {}, batch_size, description)
