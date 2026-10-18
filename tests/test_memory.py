"""Unit tests for the dialogue memory."""

import pytest

from app.services.memory import DialogueMemory, Turn


def test_add_and_get_preserves_order():
    memory = DialogueMemory(max_turns=4)
    memory.add("s", Turn(role="user", text="a"))
    memory.add("s", Turn(role="assistant", text="b"))
    assert memory.get("s") == [Turn("user", "a"), Turn("assistant", "b")]


def test_caps_history_at_max_turns():
    memory = DialogueMemory(max_turns=2)
    for i in range(3):
        memory.add("s", Turn(role="user", text=str(i)))
    assert [turn.text for turn in memory.get("s")] == ["1", "2"]


def test_sessions_are_isolated():
    memory = DialogueMemory(max_turns=4)
    memory.add("a", Turn(role="user", text="x"))
    memory.add("b", Turn(role="user", text="y"))
    assert [turn.text for turn in memory.get("a")] == ["x"]
    assert [turn.text for turn in memory.get("b")] == ["y"]


def test_reset_clears_history():
    memory = DialogueMemory(max_turns=4)
    memory.add("s", Turn(role="user", text="x"))
    memory.reset("s")
    assert memory.get("s") == []


def test_invalid_max_turns_raises():
    with pytest.raises(ValueError):
        DialogueMemory(max_turns=0)
