"""Anthropic adapter. Needs the ``anthropic`` extra: ``pip install tenet-harness[anthropic]``."""
from .base import Capabilities
from .json_only import JsonOnly


class AnthropicProvider:
    """Chat with Claude models. ``api_key=None`` reads ``ANTHROPIC_API_KEY``.

    The Messages API has no JSON mode, so ``chat(structured=True)`` asks for JSON and unwraps code fences
    (``Capabilities.json_mode`` is False). Embeddings are not provided: use another provider for routing.
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, max_tokens: int = 4096, context_window: int | None = 200_000,
                 temperature: float = 0, max_retries: int = 2):
        try:
            from langchain_anthropic import ChatAnthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError("AnthropicProvider needs the 'anthropic' extra: "
                              "pip install tenet-harness[anthropic]") from exc
        self.model, self.api_key, self.base_url = model, api_key, base_url
        self.timeout, self.max_tokens, self.temperature, self.max_retries = timeout, max_tokens, temperature, max_retries
        self.capabilities = Capabilities(tool_calling=True, streaming=True, json_mode=False,
                                         reports_usage=True, context_window=context_window)

    def chat(self, structured: bool = False):
        from langchain_anthropic import ChatAnthropic
        options: dict = {}
        if self.base_url:
            options["base_url"] = self.base_url
        if self.api_key:
            options["api_key"] = self.api_key
        model = ChatAnthropic(model=self.model, timeout=self.timeout, max_tokens=self.max_tokens,
                              temperature=self.temperature, max_retries=self.max_retries, **options)
        return JsonOnly(model) if structured else model
