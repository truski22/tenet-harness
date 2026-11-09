"""Masking secrets before arguments reach the audit log, approvals or event stream."""
import re
from typing import Any, Callable

Redactor = Callable[[dict], dict]

SECRET_KEY = re.compile(r"(token|secret|password|passwd|authorization|api[_-]?key|credential|private[_-]?key)",
                        re.IGNORECASE)
MASK = "[REDACTED]"


def default_redact(value: Any) -> Any:
    """Recursively mask values whose key looks like a credential. Returns a copy."""
    if isinstance(value, dict):
        return {key: MASK if isinstance(key, str) and SECRET_KEY.search(key) else default_redact(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [default_redact(item) for item in value]
    return value
