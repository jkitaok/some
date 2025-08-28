# Example: Adding a custom LLM provider

This example shows how to bring your own model provider by implementing `BaseLanguageModel` and registering it so the rest of the system can call it via `get_language_model(provider=...)`.

Two ways to use:

1) Call it from your own Python scripts directly.
2) Register it under a provider name and use it in your extraction pipelines.

## 1) Minimal provider implementation

```
from typing import Any, Dict, List, Optional, Tuple
from some import BaseLanguageModel

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

## 2) Registering the provider for use in extraction pipelines

Create a small module that registers your provider under a name:

```
from some import register_language_model
from .my_provider import MyLanguageModel

register_language_model("myprovider", lambda **kw: MyLanguageModel(**kw))
```

Use your provider in extraction scripts:

```python
from some.inference import get_language_model

# Use your custom provider
lm = get_language_model(provider="myprovider", model="your-model")
results, workers, timing = lm.generate(inputs)
```

This will route all LLM calls to your custom provider.
