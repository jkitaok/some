# Lightweight package init to avoid importing heavy dependencies by default
# Export names lazily to prevent requiring openai for simple operations like --help

__all__ = ["BaseLanguageModel", "register_language_model"]


def __getattr__(name):  # PEP 562 lazy export
    if name in ("BaseLanguageModel", "register_language_model"):
        from .inference import BaseLanguageModel, register_language_model
        return {"BaseLanguageModel": BaseLanguageModel, "register_language_model": register_language_model}[name]
    raise AttributeError(name)
