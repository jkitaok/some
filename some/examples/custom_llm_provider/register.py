from __future__ import annotations

from some import register_language_model
from .my_provider import MyLanguageModel

# Register under provider name "myprovider". Use it via get_language_model(provider="myprovider")
register_language_model("myprovider", lambda **kw: MyLanguageModel(**kw))
