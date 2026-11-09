import ssl
from dataclasses import dataclass
from typing import Protocol

from tenet_harness.contracts import Toolset


@dataclass(frozen=True)
class ConnectionField:
    name: str
    label: str
    secret: bool = False
    placeholder: str = ""


class Integration(Protocol):
    """External system the user connects from Settings; contributes one toolset to the agent."""
    id: str
    name: str
    fields: tuple[ConnectionField, ...]

    def status(self) -> dict: ...
    def connect(self, values: dict[str, str]) -> dict: ...
    def disconnect(self) -> None: ...
    def toolset(self) -> Toolset: ...
    def close(self) -> None: ...


def system_tls() -> ssl.SSLContext:
    """Operating-system trust store, so corporate TLS inspection certificates are honoured."""
    import truststore
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
