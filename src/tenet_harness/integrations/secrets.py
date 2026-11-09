from typing import Protocol


class SecretStore(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    def delete(self, key: str) -> None: ...


class KeyringSecrets:
    """Operating-system credential vault (Windows Credential Manager); never SQLite or .env."""

    def __init__(self, service: str):
        self.service = service

    def get(self, key: str) -> str | None:
        import keyring
        return keyring.get_password(self.service, key)

    def set(self, key: str, value: str) -> None:
        import keyring
        keyring.set_password(self.service, key, value)

    def delete(self, key: str) -> None:
        import keyring
        from keyring.errors import PasswordDeleteError
        try:
            keyring.delete_password(self.service, key)
        except PasswordDeleteError:
            pass


class MemorySecrets:
    """Process-local store for tests and schema export."""

    def __init__(self):
        self.values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)
