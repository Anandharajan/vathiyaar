from collections import deque
from typing import Deque, List, Tuple


class Memory:
    """Tiny rolling memory of previous turns."""

    def __init__(self, max_turns: int = 10) -> None:
        self._buf: Deque[Tuple[str, str]] = deque(maxlen=max_turns)

    def add(self, user: str, assistant: str) -> None:
        self._buf.append((user, assistant))

    def window(self) -> str:
        lines: List[str] = []
        for idx, (user, assistant) in enumerate(self._buf):
            lines.append(f"U{idx}: {user}")
            lines.append(f"A{idx}: {assistant}")
        return "\n".join(lines)
