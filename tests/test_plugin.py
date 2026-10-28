from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import httpx
import pytest
from nace_sdk import NaceUnprocessableEntityError, Noul, NoulAnswer

pytest.importorskip("google.adk")

from adk_nace.check import CheckError
from adk_nace.plugin import DrexPlugin
from tests.test_check import FakeClient


def _plugin(client, *, model="drex-v2.0"):
    return DrexPlugin(
        client,
        {"policy": Noul(instructions="Is this blocked?")},
        lambda _name, answer: answer.noul >= 0.5,
        model=model,
    )


def _request(*parts):
    return SimpleNamespace(contents=[SimpleNamespace(parts=list(parts))])


def _part(text=None, data=None):
    inline = None if data is None else SimpleNamespace(data=data)
    return SimpleNamespace(text=text, inline_data=inline, file_data=None)


def test_before_model_blocks_and_keeps_image_order():
    client = FakeClient({"policy": NoulAnswer(noul=0.9)})
    plugin = _plugin(client, model="drex-max")
    response = asyncio.run(
        plugin.before_model_callback(
            callback_context=None,
            llm_request=_request(_part("see this"), _part(data="second"), _part(data="first")),
        )
    )
    assert response is not None
    assert response.content.parts[0].text == "Drex (drex-max) blocked this request: policy"
    assert client.calls[0]["images"] == ["second", "first"]
    assert client.calls[0]["model"] == "drex-max"
    assert client.calls[0]["state"] == "see this"


def test_before_model_allow_returns_none():
    client = FakeClient({"policy": NoulAnswer(noul=0.1)})
    plugin = _plugin(client)
    response = asyncio.run(
        plugin.before_model_callback(
            callback_context=None,
            llm_request=_request(_part("fine")),
        )
    )
    assert response is None


def test_before_model_422_is_not_allow():
    error = NaceUnprocessableEntityError(
        422,
        {"issues": ["bad"]},
        httpx.Headers({}),
        message="unprocessable",
    )
    plugin = _plugin(FakeClient(error=error))
    with pytest.raises(NaceUnprocessableEntityError):
        asyncio.run(
            plugin.before_model_callback(
                callback_context=None,
                llm_request=_request(_part("bad")),
            )
        )


def test_after_model_replaces_a_blocked_response():
    client = FakeClient({"policy": NoulAnswer(noul=0.9)})
    plugin = _plugin(client)
    original = SimpleNamespace(content=SimpleNamespace(parts=[_part("model said this")]))
    response = asyncio.run(
        plugin.after_model_callback(callback_context=None, llm_response=original)
    )
    assert response is not None
    assert "blocked" in response.content.parts[0].text
    assert client.calls[0]["state"] == "model said this"
    assert client.calls[0]["images"] is None


def test_before_tool_block_and_allow():
    blocked = _plugin(FakeClient({"policy": NoulAnswer(noul=0.9)}))
    tool = SimpleNamespace(name="lookup")
    result = asyncio.run(
        blocked.before_tool_callback(tool=tool, tool_args={"q": "x"}, tool_context=None)
    )
    assert result == {
        "blocked": True,
        "message": "Drex (drex-v2.0) blocked this request: policy",
    }

    allowed = _plugin(FakeClient({"policy": NoulAnswer(noul=0.1)}))
    assert (
        asyncio.run(allowed.before_tool_callback(tool=tool, tool_args={"q": "x"}, tool_context=None))
        is None
    )


def test_empty_turn_still_calls_system_one():
    client = FakeClient({"policy": NoulAnswer(noul=0.1)})
    plugin = _plugin(client)
    response = asyncio.run(
        plugin.before_model_callback(
            callback_context=None,
            llm_request=SimpleNamespace(contents=None),
        )
    )
    assert response is None
    assert client.calls == [{"state": "", "questions": client.calls[0]["questions"], "model": "drex-v2.0", "images": None, "kwargs": {}}]


def test_real_genai_parts_keep_byte_images_in_order():
    from google.genai import types
    from nace_sdk import image_input

    from tests.test_check import PNG

    jpeg = base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAb/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAGf/9k="
    )
    expected = [image_input(PNG), image_input(jpeg)]
    client = FakeClient({"policy": NoulAnswer(noul=0.1)})
    plugin = _plugin(client, model="drex-max")
    parts = [
        types.Part.from_text(text="look"),
        types.Part.from_bytes(data=PNG, mime_type="image/png"),
        types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
    ]
    asyncio.run(
        plugin.before_model_callback(
            callback_context=None,
            llm_request=SimpleNamespace(contents=[types.Content(role="user", parts=parts)]),
        )
    )
    assert client.calls[0]["state"] == "look"
    assert client.calls[0]["images"] == expected

    client.calls.clear()
    flipped = [parts[2], parts[0], parts[1]]
    asyncio.run(
        plugin.before_model_callback(
            callback_context=None,
            llm_request=SimpleNamespace(contents=[types.Content(role="user", parts=flipped)]),
        )
    )
    assert client.calls[0]["images"] == [expected[1], expected[0]]


def test_missing_answer_is_not_allow_from_the_plugin():
    plugin = _plugin(FakeClient({}))
    with pytest.raises(CheckError):
        asyncio.run(
            plugin.before_model_callback(
                callback_context=None,
                llm_request=_request(_part("text")),
            )
        )
