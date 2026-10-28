"""Google ADK adapter. Importing this module requires google.adk."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from adk_nace.check import (
    DEFAULT_MODEL,
    FailIf,
    Question,
    block_message,
    check,
    read_contents,
    tool_refusal,
    tool_state,
)
from nace_sdk import NaceClient


class DrexPlugin(BasePlugin):
    """Block model and tool calls when a Drex check fails.

    `None` allows the call. A returned response skips the model call or replaces
    the model output. A returned dict skips the tool. A 422 or missing answer
    propagates; it is not treated as an allow.
    """

    def __init__(
        self,
        client: NaceClient,
        questions: Mapping[str, Question],
        fail_if: FailIf,
        *,
        model: str = DEFAULT_MODEL,
        name: str = "drex",
    ) -> None:
        super().__init__(name=name)
        self.client = client
        self.questions = questions
        self.fail_if = fail_if
        self.model = model

    async def before_model_callback(
        self,
        *,
        callback_context: Any,
        llm_request: Any,
    ) -> LlmResponse | None:
        text, images = read_contents(llm_request.contents)
        return _model_refusal(self._decide(text, images or None))

    async def after_model_callback(
        self,
        *,
        callback_context: Any,
        llm_response: Any,
    ) -> LlmResponse | None:
        content = llm_response.content
        text, images = read_contents(() if content is None else (content,))
        return _model_refusal(self._decide(text, images or None))

    async def before_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
    ) -> dict[str, Any] | None:
        return tool_refusal(self._decide(tool_state(getattr(tool, "name", ""), tool_args)))

    def _decide(self, state: str, images: list[str] | None = None):
        return check(
            self.client,
            state,
            self.questions,
            fail_if=self.fail_if,
            model=self.model,
            images=images,
        )


def _model_refusal(decision) -> LlmResponse | None:
    message = block_message(decision)
    if message is None:
        return None
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )
    )
