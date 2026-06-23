"""Unit tests for LLM adapters (offline — no Ollama required)."""

from sdk.nura.adapters.ollama_adapter import OllamaAdapter


class TestOllamaAdapter:
    def test_default_model(self):
        adapter = OllamaAdapter()
        assert adapter._model == "llama3.2:1b"

    def test_custom_model(self):
        adapter = OllamaAdapter(model="llama3.2:3b")
        assert adapter._model == "llama3.2:3b"

    def test_context_window_default(self):
        adapter = OllamaAdapter()
        assert adapter.context_window == 4096

    def test_context_window_custom(self):
        adapter = OllamaAdapter(context_window_size=8192)
        assert adapter.context_window == 8192

    def test_supports_logprobs(self):
        assert OllamaAdapter().supports_logprobs is True

    def test_count_tokens_non_zero(self):
        adapter = OllamaAdapter()
        assert adapter.count_tokens("Hello, world!") >= 1

    def test_count_tokens_empty_string(self):
        adapter = OllamaAdapter()
        assert adapter.count_tokens("") == 1

    def test_count_tokens_scales_with_length(self):
        adapter = OllamaAdapter()
        short = adapter.count_tokens("Hi")
        long = adapter.count_tokens("Hi " * 100)
        assert long > short

    def test_count_tokens_is_integer(self):
        adapter = OllamaAdapter()
        result = adapter.count_tokens("Some text here")
        assert isinstance(result, int)
