# Example: Adding a custom LLM provider

This example shows how to bring your own model provider by implementing `BaseLanguageModel` and registering it so the rest of the system can call it via `get_language_model(provider=...)`.

Two ways to use:

1) Call it from your own Python scripts directly.
2) Register it under a provider name and use it from the CLI by setting `PAPERSCRAPER_PLUGINS`.

## 1) Minimal provider implementation

```
from typing import Any, Dict, List, Optional, Tuple
from paperscraper import BaseLanguageModel

class MyLanguageModel(BaseLanguageModel):
    def __init__(self, *, model: Optional[str] = None, **kwargs):
        super().__init__(model=model)
        # init client(s) here

    def generate(self, inputs: List[Dict[str, Any]], *, max_workers: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
        results = []
        for payload in inputs:
            # Implement your call here
            # You should respect payload["result_key"] when building the result
            result_key = payload.get("result_key", "result")
            results.append({
                "input_tokens": 0,
                "output_tokens": 0,
                result_key: {"demo": True},
            })
        return results, 1
```

## 2) Registering the provider for CLI use

Create a small module that registers your provider under a name. Then load it at runtime by setting `PAPERSCRAPER_PLUGINS`.

```
from paperscraper import register_language_model
from .my_provider import MyLanguageModel

register_language_model("myprovider", lambda **kw: MyLanguageModel(**kw))
```

Run the CLI with your provider:

```
PAPERSCRAPER_PLUGINS=paperscraper.examples.custom_llm_provider.register \
  paperscraper collect cs.AI 2024-01-15 --steps affiliations --provider myprovider -y --num 3
```

This will route all LLM calls in the affiliations step to your provider.
