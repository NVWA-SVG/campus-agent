"""按会话保存有限长度的对话历史。"""

from __future__ import annotations

import threading
from collections import OrderedDict

from campus_agent.domain import Message


class ConversationMemory:
    def __init__(self, max_messages: int = 20, max_sessions: int = 256) -> None:
        if max_messages < 2:
            raise ValueError("max_messages 必须至少为 2")
        if max_sessions < 1:
            raise ValueError("max_sessions 必须至少为 1")
        self._max_messages = max_messages
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[Message]] = OrderedDict()
        self._lock = threading.RLock()

    def add(self, session_id: str, message: Message) -> None:
        with self._lock:
            self._append_locked(session_id, (message,))

    def add_turn(
        self,
        session_id: str,
        user_message: Message,
        assistant_message: Message,
    ) -> None:
        """在同一个锁内提交完整回合，避免并发读取到半个回合。"""

        with self._lock:
            self._append_locked(session_id, (user_message, assistant_message))

    def get(self, session_id: str) -> tuple[Message, ...]:
        with self._lock:
            messages = self._sessions.get(session_id)
            if messages is None:
                return ()
            self._sessions.move_to_end(session_id)
            return tuple(messages)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _append_locked(
        self,
        session_id: str,
        new_messages: tuple[Message, ...],
    ) -> None:
        messages = self._sessions.get(session_id)
        if messages is None:
            if len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            messages = []
            self._sessions[session_id] = messages
        else:
            self._sessions.move_to_end(session_id)
        messages.extend(new_messages)
        if len(messages) > self._max_messages:
            del messages[: len(messages) - self._max_messages]
