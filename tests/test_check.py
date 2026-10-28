from __future__ import annotations

import base64
from types import SimpleNamespace

import httpx
import pytest
from nace_sdk import (
    Choice,
    ChoiceAnswer,
    NaceUnprocessableEntityError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)

from adk_nace import CheckError, check, open_client
from adk_nace.check import (
    CLIENT_INFO,
    block_message,
    read_contents,
    tool_refusal,
    tool_state,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class FakeClient:
    def __init__(self, answers=None, error=None):
        self.answers = {} if answers is None else answers
        self.error = error
        self.calls = []

    def system_one(self, state, questions, *, model=None, images=None, **kwargs):
        self.calls.append(
            {
                "state": state,
                "questions": dict(questions),
                "model": model,
                "images": None if images is None else list(images),
                "kwargs": kwargs,
            }
        )
        if self.error is not None:
            raise self.error
        return SimpleNamespace(answers=self.answers, model=model)


def fail_if(_name, answer):
    if answer.type == "noul":
        return answer.noul >= 0.5
    if answer.type == "choice":
        return answer.choice == "block"
    if answer.type == "score":
        return answer.score >= 3
    raise AssertionError(answer.type)


def pair(kind, *, block):
    if kind == "noul":
        return {"q": Noul(instructions="Is it blocked?")}, {
            "q": NoulAnswer(noul=0.9 if block else 0.1)
        }
    if kind == "choice":
        label = "block" if block else "ok"
        other = "ok" if block else "block"
        return {"route": Choice(criteria={"block": "blocked", "ok": "allowed"})}, {
            "route": ChoiceAnswer(
                choice=label,
                confidence=0.8,
                probabilities={label: 0.8, other: 0.2},
            )
        }
    return {"severity": Score(criteria=["none", "low", "mid", "high"])}, {
        "severity": ScoreAnswer(
            score=4 if block else 1,
            confidence=0.7,
            legend=("none", "low", "mid", "high"),
            probabilities={1: 0.2, 4: 0.8},
        )
    }


@pytest.mark.parametrize("model", ["drex-v2.0", "drex-max"])
@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
@pytest.mark.parametrize("block", [False, True])
def test_question_types_on_both_models(model, kind, block):
    questions, answers = pair(kind, block=block)
    client = FakeClient(answers)
    decision = check(client, "state", questions, fail_if=fail_if, model=model)
    assert client.calls[0]["model"] == model
    sent = next(iter(client.calls[0]["questions"].values()))
    assert sent.type == kind
    assert decision.blocked is block
    assert decision.model == model
    assert (decision.hits == (next(iter(questions)),)) is block


def test_default_model_is_sent_explicitly():
    client = FakeClient({"q": NoulAnswer(noul=0.1)})
    check(client, "state", {"q": Noul(instructions="x")}, fail_if=fail_if)
    assert client.calls[0]["model"] == "drex-v2.0"


def test_padded_model_is_stripped_before_the_call():
    client = FakeClient({"q": NoulAnswer(noul=0.1)})
    check(client, "state", {"q": Noul(instructions="x")}, fail_if=fail_if, model="  drex-v2.0  ")
    assert client.calls[0]["model"] == "drex-v2.0"


@pytest.mark.parametrize("model", ["", "   ", "\n"])
def test_empty_model_does_not_call(model):
    client = FakeClient({})
    with pytest.raises(CheckError):
        check(client, "state", {"q": Noul(instructions="x")}, fail_if=fail_if, model=model)
    assert client.calls == []


@pytest.mark.parametrize("images", [["only"], ["first", "second"], ["second", "first"]])
def test_image_order_on_drex_max(images):
    client = FakeClient({"q": NoulAnswer(noul=0.1)})
    check(
        client,
        "see",
        {"q": Noul(instructions="x")},
        fail_if=fail_if,
        model="drex-max",
        images=images,
    )
    assert client.calls[0]["images"] == images


def test_eight_images_are_sent_in_order():
    images = [f"img-{i}" for i in range(8)]
    client = FakeClient({"q": NoulAnswer(noul=0.1)})
    check(
        client,
        "see",
        {"q": Noul(instructions="x")},
        fail_if=fail_if,
        model="drex-max",
        images=images,
    )
    assert client.calls[0]["images"] == images


def test_nine_images_do_not_call():
    client = FakeClient({})
    with pytest.raises(CheckError):
        check(
            client,
            "see",
            {"q": Noul(instructions="x")},
            fail_if=fail_if,
            model="drex-max",
            images=[str(i) for i in range(9)],
        )
    assert client.calls == []


def test_images_on_drex_v2_do_not_call():
    client = FakeClient({})
    with pytest.raises(CheckError):
        check(
            client,
            "see",
            {"q": Noul(instructions="x")},
            fail_if=fail_if,
            model="drex-v2.0",
            images=["one"],
        )
    assert client.calls == []


def test_image_string_is_not_iterated():
    client = FakeClient({})
    with pytest.raises(CheckError):
        check(
            client,
            "see",
            {"q": Noul(instructions="x")},
            fail_if=fail_if,
            model="drex-max",
            images="abcd",
        )
    assert client.calls == []


def test_drex_max_question_cap():
    allowed = {f"q{i}": Noul(instructions="x") for i in range(16)}
    client = FakeClient({name: NoulAnswer(noul=0.1) for name in allowed})
    check(client, "s", allowed, fail_if=fail_if, model="drex-max")
    assert len(client.calls) == 1

    rejected = {f"q{i}": Noul(instructions="x") for i in range(17)}
    client = FakeClient({})
    with pytest.raises(CheckError):
        check(client, "s", rejected, fail_if=fail_if, model="drex-max")
    assert client.calls == []


def test_drex_v2_question_cap():
    allowed = {f"q{i}": Noul(instructions="x") for i in range(512)}
    client = FakeClient({name: NoulAnswer(noul=0.1) for name in allowed})
    check(client, "s", allowed, fail_if=fail_if, model="drex-v2.0")
    assert len(client.calls) == 1

    rejected = {f"q{i}": Noul(instructions="x") for i in range(513)}
    client = FakeClient({})
    with pytest.raises(CheckError):
        check(client, "s", rejected, fail_if=fail_if, model="drex-v2.0")
    assert client.calls == []


def test_unknown_model_is_not_capped():
    questions = {f"q{i}": Noul(instructions="x") for i in range(17)}
    client = FakeClient({name: NoulAnswer(noul=0.1) for name in questions})
    check(client, "s", questions, fail_if=fail_if, model="drex-custom")
    assert client.calls[0]["model"] == "drex-custom"
    assert len(client.calls[0]["questions"]) == 17


def test_422_is_not_a_pass():
    error = NaceUnprocessableEntityError(
        422,
        {"issues": ["bad"]},
        httpx.Headers({}),
        message="unprocessable",
    )
    client = FakeClient(error=error)
    with pytest.raises(NaceUnprocessableEntityError):
        check(client, "s", {"q": Noul(instructions="x")}, fail_if=fail_if)
    assert len(client.calls) == 1


def test_other_api_error_is_not_a_pass():
    client = FakeClient(error=RuntimeError("down"))
    with pytest.raises(RuntimeError, match="down"):
        check(client, "s", {"q": Noul(instructions="x")}, fail_if=fail_if)


def test_missing_answer_is_not_allow():
    client = FakeClient({})
    with pytest.raises(CheckError, match="missing an answer"):
        check(client, "s", {"q": Noul(instructions="x")}, fail_if=fail_if)
    assert len(client.calls) == 1


def test_none_instructions_are_sent_as_empty_string():
    original = Noul()
    assert original.instructions is None
    client = FakeClient({"q": NoulAnswer(noul=0.1)})
    check(client, "s", {"q": original}, fail_if=fail_if)
    assert original.instructions is None
    assert client.calls[0]["questions"]["q"].instructions == ""


def test_open_client_retry_budget():
    client = open_client(api_key="sk-test")
    assert client._config.default_headers["X-Nace-Client"] == CLIENT_INFO
    timeout = client._config.timeout
    assert float(getattr(timeout, "read", timeout)) == 60
    assert client._retry.max_retries != 0
    assert client._retry.timeout == 180
    assert 429 in client._retry.http_statuses
    assert 529 in client._retry.http_statuses


def test_package_import_does_not_load_adk():
    import sys

    sys.modules.pop("google.adk", None)
    import adk_nace

    assert not hasattr(adk_nace, "DrexPlugin")
    assert "google.adk" not in sys.modules


def test_plugin_import_is_separate():
    import importlib

    try:
        importlib.import_module("google.adk")
    except ImportError:
        with pytest.raises(ImportError):
            importlib.import_module("adk_nace.plugin")
    else:
        importlib.import_module("adk_nace.plugin")


def test_read_contents_keeps_image_order():
    first = SimpleNamespace(text="hello", inline_data=SimpleNamespace(data="data:image/png;base64,YQ=="))
    second = SimpleNamespace(text=None, inline_data=SimpleNamespace(data=PNG))
    flipped = SimpleNamespace(parts=[second, first])
    forward = SimpleNamespace(parts=[first, second])
    text, images = read_contents([forward])
    assert text == "hello"
    assert images[0] == "data:image/png;base64,YQ=="
    assert images[1].startswith("data:image/png;base64,")
    _, flipped_images = read_contents([flipped])
    assert flipped_images == [images[1], images[0]]


def test_file_image_is_not_dropped():
    part = SimpleNamespace(text=None, inline_data=None, file_data=SimpleNamespace(file_uri="https://example.test/a.png"))
    with pytest.raises(CheckError, match="not inlined"):
        read_contents([SimpleNamespace(parts=[part])])


def test_block_message_and_tool_refusal():
    from adk_nace import CheckDecision

    allowed = CheckDecision(model="drex-v2.0", answers={}, hits=())
    blocked = CheckDecision(model="drex-max", answers={}, hits=("policy", "severity"))
    assert block_message(allowed) is None
    assert tool_refusal(allowed) is None
    assert block_message(blocked) == "Drex (drex-max) blocked this request: policy, severity"
    assert tool_refusal(blocked) == {
        "blocked": True,
        "message": "Drex (drex-max) blocked this request: policy, severity",
    }


def test_tool_state_is_stable():
    assert tool_state("lookup", {"b": 1, "a": 2}) == tool_state("lookup", {"a": 2, "b": 1})
