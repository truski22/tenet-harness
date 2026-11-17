"""OpenAI and OpenAI-compatible adapter (OpenAI, Azure-style gateways, vLLM, LM Studio, OpenRouter...).

Needs the ``openai`` extra: ``pip install tenet-harness[openai]``.
"""
from .base import Capabilities


class OpenAICompatibleProvider:
    """Chat through any server that speaks the OpenAI chat-completions API.

    ``base_url=None`` targets OpenAI itself. ``api_key=None`` reads ``OPENAI_API_KEY``; local servers that
    ignore the key still need a non-empty value, so pass any string. Embeddings are not provided.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 120.0, context_window: int | None = None, temperature: float = 0,
                 json_mode: bool = True, reports_usage: bool = True, max_retries: int = 2,
                 default_headers: dict[str, str] | None = None):
        try:
            from langchain_openai import ChatOpenAI  # noqa: F401
        except ImportError as exc:
            raise ImportError("OpenAICompatibleProvider needs the 'openai' extra: "
                              "pip install tenet-harness[openai]") from exc
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self.timeout, self.temperature, self.max_retries = timeout, temperature, max_retries
        self.default_headers = default_headers
        self.capabilities = Capabilities(tool_calling=True, streaming=True, json_mode=json_mode,
                                         reports_usage=reports_usage, context_window=context_window)

    def chat(self, structured: bool = False):
        from langchain_openai import ChatOpenAI
        options: dict = {}
        if structured and self.capabilities.json_mode:
            options["model_kwargs"] = {"response_format": {"type": "json_object"}}
        model = ChatOpenAI(model=self.model, base_url=self.base_url, api_key=self.api_key,
                           timeout=self.timeout, temperature=self.temperature, max_retries=self.max_retries,
                           stream_usage=self.capabilities.reports_usage, default_headers=self.default_headers,
                           **options)
        if structured and not self.capabilities.json_mode:
            from .json_only import JsonOnly
            return JsonOnly(model)
        return model
