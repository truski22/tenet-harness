"""Ollama adapter (local inference and embeddings)."""
from langchain_ollama import ChatOllama, OllamaEmbeddings


class OllamaProvider:
    def __init__(self, model: str, url: str, timeout: float, context_window: int = 8192,
                 embedding_model: str | None = None):
        self.model = model
        self.url = url
        self.timeout = timeout
        self.context_window = context_window
        self.embedding_model = embedding_model
        self._embeddings: OllamaEmbeddings | None = None

    def chat(self, structured: bool = False):
        return ChatOllama(
            model=self.model, base_url=self.url, temperature=0, num_ctx=self.context_window,
            client_kwargs={"timeout": self.timeout, "trust_env": False},
            format="json" if structured else None,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.embedding_model:
            raise ValueError("No embedding model configured.")
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(model=self.embedding_model, base_url=self.url,
                                                client_kwargs={"timeout": self.timeout, "trust_env": False})
        return self._embeddings.embed_documents(texts)
