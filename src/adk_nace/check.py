"""Pure Drex check. This module does not import Google ADK."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from nace_sdk import Choice, NaceClient, Noul, RetryPolicy, Score, image_input

CLIENT_INFO = "adk-nace/0.1.0"
DEFAULT_MODEL = "drex-v2.0"
MAX_IMAGES = 8
MAX_QUESTIONS = {
    "drex-max": 16,
    "drex-v2.0": 512,
}

Question = Noul | Choice | Score
FailIf = Callable[[str, Any], bool]


class CheckError(Exception):
    """The check cannot decide from the local inputs or the response."""


@dataclass(frozen=True)
class CheckDecision:
    """A completed check. `blocked` is true exactly when `hits` is non-empty."""

    model: str
    answers: Mapping[str, Any]
    hits: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        return bool(self.hits)


def open_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> NaceClient:
    """Client with this package's name, a 60s request timeout, and a 180s retry budget."""
    return NaceClient(
        api_key=api_key,
        base_url=base_url,
        client_info=CLIENT_INFO,
        timeout=60,
        retry=RetryPolicy(timeout=180),
    )


def check(
    client: NaceClient,
    state: str,
    questions: Mapping[str, Question],
    *,
    fail_if: FailIf,
    model: str = DEFAULT_MODEL,
    images: Sequence[str] | None = None,
) -> CheckDecision:
    """Run one Drex check.

    Raises `CheckError` before the call when the model, question count, or images are
    not allowed. A missing answer raises `CheckError` after the call. SDK errors,
    including HTTP 422, propagate unchanged.
    """
    model_id = _model(model)
    prepared = _questions(questions, model_id)
    image_list = _images(images, model_id)
    response = client.system_one(state, prepared, images=image_list, model=model_id)
    answers = _answers(prepared, response)
    hits = tuple(name for name, answer in answers.items() if fail_if(name, answer))
    return CheckDecision(model=model_id, answers=MappingProxyType(answers), hits=hits)


def block_message(decision: CheckDecision) -> str | None:
    """Refusal text for a blocked check, or None when the check allowed the text."""
    if not decision.hits:
        return None
    return f"Drex ({decision.model}) blocked this request: {', '.join(decision.hits)}"


def tool_refusal(decision: CheckDecision) -> dict[str, Any] | None:
    """Tool-callback result. None allows the tool; a dict skips it."""
    message = block_message(decision)
    if message is None:
        return None
    return {"blocked": True, "message": message}


def read_contents(contents: Sequence[Any] | None) -> tuple[str, list[str]]:
    """Join part text and collect inline images in encounter order.

    File URIs are not fetched. A part that only references a file raises `CheckError`
    so that image is not dropped.
    """
    texts: list[str] = []
    images: list[str] = []
    for content in contents or ():
        for part in getattr(content, "parts", None) or ():
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
            image = _image_from_part(part)
            if image is not None:
                images.append(image)
    return "\n".join(texts), images


def tool_state(name: str, args: Mapping[str, Any] | None) -> str:
    """Stable text for a tool call. Key order does not change the check."""
    payload = json.dumps(args or {}, default=str, sort_keys=True)
    return f"{name}\n{payload}"


def _model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise CheckError("model must be a non-empty string")
    return model.strip()


def _questions(questions: Mapping[str, Question], model: str) -> dict[str, Question]:
    if not isinstance(questions, Mapping) or not questions:
        raise CheckError("at least one question is required")
    limit = MAX_QUESTIONS.get(model)
    if limit is not None and len(questions) > limit:
        raise CheckError(f"{model} accepts at most {limit} questions")
    prepared: dict[str, Question] = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name:
            raise CheckError("question names must be non-empty strings")
        if getattr(question, "instructions", "") is None and hasattr(question, "model_copy"):
            question = question.model_copy(update={"instructions": ""})
        prepared[name] = question
    return prepared


def _images(images: Sequence[str] | None, model: str) -> list[str] | None:
    if images is None:
        return None
    if isinstance(images, (str, bytes)):
        raise CheckError("images must be a list")
    image_list = list(images)
    if not image_list:
        return None
    if model != "drex-max":
        raise CheckError(f"images are only supported on drex-max, not {model}")
    if len(image_list) > MAX_IMAGES:
        raise CheckError(f"drex-max accepts at most {MAX_IMAGES} images")
    return image_list


def _answers(questions: Mapping[str, Question], response: Any) -> dict[str, Any]:
    raw = getattr(response, "answers", None)
    if not isinstance(raw, Mapping):
        raise CheckError("Drex response did not include answers")
    answers: dict[str, Any] = {}
    for name in questions:
        if name not in raw or raw[name] is None:
            raise CheckError(f"Drex response is missing an answer for {name!r}")
        answers[name] = raw[name]
    return answers


def _image_from_part(part: Any) -> str | None:
    blob = getattr(part, "inline_data", None)
    if blob is not None:
        data = getattr(blob, "data", None)
        if not data:
            raise CheckError("inline image has no data")
        return _as_image(data)
    file_uri = getattr(getattr(part, "file_data", None), "file_uri", None)
    if file_uri:
        raise CheckError("file image references are not inlined; pass image bytes or a data URL")
    return None


def _as_image(data: Any) -> str:
    if isinstance(data, bytes):
        converted = image_input(data)
        if not isinstance(converted, str):
            raise CheckError("inline image bytes did not become a data URL")
        return converted
    if isinstance(data, str):
        return data
    raise CheckError("inline image data must be bytes or a string")
