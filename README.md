# adk-nace

A Drex check for Google ADK. The check calls `NaceClient.system_one` from the published `nace-sdk`. It does not send its own HTTP requests.

```python
from nace_sdk import Noul, Choice, Score
from adk_nace import check, open_client

client = open_client()  # client_info adk-nace/0.1.0, timeout 60s, retry budget 180s
decision = check(
    client,
    "user text",
    {
        "policy": Noul(instructions="Is this a policy violation?"),
        "route": Choice(criteria={"allow": "allowed", "block": "blocked"}),
        "severity": Score(criteria=["none", "low", "high"]),
    },
    fail_if=lambda name, answer: (
        answer.noul >= 0.5 if answer.type == "noul"
        else answer.choice == "block" if answer.type == "choice"
        else answer.score >= 2
    ),
)
```

`decision.blocked` is true when `fail_if` returns true for any question. The default model is `drex-v2.0`, passed on the call. Model ids are open strings. This package does not send `drex-latest` unless you pass it.

Images are a list, only on `drex-max`, at most 8, in the order you pass them. Any other model with images raises `CheckError` before the call. `drex-max` rejects more than 16 questions. `drex-v2.0` rejects more than 512. Other model ids are not capped here.

A missing answer raises `CheckError`. HTTP 422 and other SDK errors propagate. Neither is an allow.

`from adk_nace import check` does not import Google ADK. The plugin is a separate module and needs `google-adk`:

```python
from adk_nace.plugin import DrexPlugin

plugin = DrexPlugin(client, questions, fail_if)
# app = App(name="guarded", root_agent=agent, plugins=[plugin])
```

`before_model_callback` and `after_model_callback` return an `LlmResponse` to block and `None` to allow. `before_tool_callback` returns a dict to skip the tool and `None` to allow it. An API error is raised, not returned as `None`.
