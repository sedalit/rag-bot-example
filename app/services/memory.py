"""In-memory per-session dialogue history."""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One utterance in a dialogue."""

    role: str  # "user" | "assistant"
    text: str


class DialogueMemory:
    """Keeps the last ``max_turns`` utterances per session."""

    def __init__(self, max_turns: int) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self._max_turns = max_turns
        self._history: dict[str, deque[Turn]] = {}

    def add(self, session_id: str, turn: Turn) -> None:
        """Append a turn, evicting the oldest when over the cap."""
        queue = self._history.get(session_id)
        if queue is None:
            queue = deque()
            self._history[session_id] = queue
        queue.append(turn)
        while len(queue) > self._max_turns:
            queue.popleft()

    def get(self, session_id: str) -> list[Turn]:
        """Return the recent turns for a session, oldest first."""
        queue = self._history.get(session_id)
        return list(queue) if queue is not None else []

    def reset(self, session_id: str) -> None:
        """Drop all history for a session."""
        self._history.pop(session_id, None)
