from __future__ import annotations

import unittest

from campus_agent.domain import Message


class MessageTests(unittest.TestCase):
    """锁定 Message 的领域约束和存储边界转换。"""

    def test_round_trip_through_serializable_payload(self) -> None:
        message = Message(role="user", content="周三有什么课程？")

        payload = message.as_dict()
        restored = Message.from_dict(payload)

        self.assertEqual(
            payload,
            {"role": "user", "content": "周三有什么课程？"},
        )
        self.assertEqual(restored, message)

    def test_rejects_roles_that_do_not_belong_in_conversation_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持的消息角色"):
            Message(role="tool", content="工具结果")  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "不支持的消息角色"):
            Message.from_dict({"role": "system", "content": "系统提示词"})

    def test_rejects_invalid_content(self) -> None:
        with self.assertRaisesRegex(TypeError, "消息内容必须是字符串"):
            Message(role="user", content=123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
