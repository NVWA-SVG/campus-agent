"""离线规划器：把自然语言请求转换为结构化工具调用。"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from campus_agent.domain import Message, Plan, ToolCall, ToolResult
from campus_agent.tools.course import WEEKDAY_ALIASES


class Planner(Protocol):
    def plan(
        self,
        query: str,
        history: Sequence[Message],
        tool_descriptions: Sequence[dict[str, object]],
    ) -> Plan: ...

    def compose(self, query: str, results: Sequence[ToolResult]) -> str: ...


class RuleBasedPlanner:
    """用于离线 MVP 的可替换规划器。

    它让项目在没有模型服务时也能验证 Agent 编排、工具协议、记忆和异常边界。
    """

    _course_words = ("课表", "上课", "什么课", "在哪里上", "教室", "几点上课")
    _knowledge_process_words = (
        "流程",
        "指南",
        "办法",
        "规定",
        "政策",
        "材料",
        "申请",
        "办理",
        "退选",
    )
    _knowledge_words = (
        "图书馆",
        "借书",
        "借阅",
        "还书",
        "成绩单",
        "校园卡",
        "饭卡",
        "挂失",
        "补办",
        "奖学金",
        "评优",
        "实验室",
        "预约",
    )
    _course_keywords = ("Python", "机器学习", "人工智能", "英语")
    _multi_intent_words = ("并且", "以及", "同时", "另外", "还要", "还想", "和")
    _live_status_words = (
        "现在",
        "当前",
        "今天",
        "开门",
        "营业",
        "开放吗",
        "排队",
        "忙不忙",
        "还能办",
        "窗口状态",
    )
    _knowledge_action_words = (
        "怎么办",
        "丢了",
        "遗失",
        "不见了",
        "挂失",
        "补办",
        "借书",
        "还书",
        "续借",
        "申请",
        "预约",
        "材料",
        "流程",
    )
    _service_aliases = (
        ("校园卡服务中心", "campus_card"),
        ("校园卡", "campus_card"),
        ("一卡通", "campus_card"),
        ("饭卡", "campus_card"),
        ("教务服务中心", "registrar"),
        ("教务处", "registrar"),
        ("教务", "registrar"),
        ("图书馆服务台", "library"),
        ("图书馆", "library"),
        ("学生事务中心", "student_affairs"),
        ("学生事务", "student_affairs"),
        ("学生处", "student_affairs"),
    )

    def plan(
        self,
        query: str,
        history: Sequence[Message],
        tool_descriptions: Sequence[dict[str, object]],
    ) -> Plan:
        cleaned_query = query.strip()
        if not cleaned_query:
            return Plan(summary="输入为空", direct_answer="请输入你想查询的校园事务。")

        available_names = {
            str(description.get("name")) for description in tool_descriptions
        }
        previous_user_text = " ".join(
            message.content for message in history[-6:] if message.role == "user"
        )
        context = f"{previous_user_text} {cleaned_query}".strip()
        calls: list[ToolCall] = []

        weekday = self._extract_weekday(cleaned_query)
        course_intent = any(word in cleaned_query for word in self._course_words)
        if weekday is not None and any(word in cleaned_query for word in ("课", "课程")):
            course_intent = True
        if not course_intent and any(word in cleaned_query for word in ("哪里", "几点")):
            course_intent = any(
                word in previous_user_text
                for word in (*self._course_words, "课程")
            )

        if weekday is None and course_intent:
            weekday = self._extract_weekday(previous_user_text)

        if course_intent and "query_courses" in available_names:
            if weekday is None:
                return Plan(
                    summary="课程查询缺少星期参数",
                    direct_answer="你想查询星期几的课程？例如：周三有什么课？",
                )
            keyword = next(
                (item for item in self._course_keywords if item.lower() in context.lower()),
                "",
            )
            calls.append(
                ToolCall(
                    call_id=f"call-{len(calls) + 1}",
                    name="query_courses",
                    arguments={"weekday": weekday, "keyword": keyword},
                )
            )

        live_status_intent = any(
            word in cleaned_query for word in self._live_status_words
        )
        if live_status_intent:
            service_code = self._extract_service_code(cleaned_query)
            if service_code is None:
                service_code = self._extract_service_code(previous_user_text)
            if service_code is None:
                return Plan(
                    summary="实时服务查询缺少服务名称",
                    direct_answer=(
                        "你想查询哪个服务当前是否开放？例如："
                        "校园卡服务中心现在开门吗？"
                    ),
                )
            if "query_campus_service_status" in available_names:
                calls.append(
                    ToolCall(
                        call_id=f"call-{len(calls) + 1}",
                        name="query_campus_service_status",
                        arguments={
                            "service_code": service_code,
                            "campus": self._extract_campus(context),
                        },
                    )
                )

        process_intent = any(
            word in cleaned_query for word in self._knowledge_process_words
        )
        process_intent = process_intent or any(
            word in cleaned_query for word in self._knowledge_action_words
        )
        knowledge_subject_intent = any(
            word in cleaned_query for word in self._knowledge_words
        )
        mixed_course_intent = course_intent and any(
            word in cleaned_query for word in self._multi_intent_words
        )
        knowledge_intent = process_intent or (
            knowledge_subject_intent and not live_status_intent
        ) or (
            mixed_course_intent and not live_status_intent
        )
        # 动态上传资料的主题无法预先全部写进关键词表。非课程问题默认尝试
        # 本地RAG；实时状态问题除外，避免用静态文档冒充当前业务数据。
        if not course_intent and not live_status_intent:
            knowledge_intent = True
        if knowledge_intent and "search_campus_knowledge" in available_names:
            knowledge_query = self._knowledge_only_query(cleaned_query)
            calls.append(
                ToolCall(
                    call_id=f"call-{len(calls) + 1}",
                    name="search_campus_knowledge",
                    arguments={"query": knowledge_query},
                )
            )

        if not calls:
            return Plan(
                summary="没有匹配到合适的工具",
                direct_answer=(
                    "我目前可以查询课程、图书馆借阅、成绩单、校园卡、奖学金和实验室预约流程。"
                    "你可以问：周三有什么课？"
                ),
            )

        names = "、".join(call.name for call in calls)
        return Plan(summary=f"准备调用：{names}", tool_calls=tuple(calls))

    def compose(self, query: str, results: Sequence[ToolResult]) -> str:
        if not results:
            return "本次请求没有产生工具结果。"

        sections: list[str] = []
        for result in results:
            if result.ok:
                if (
                    len(results) > 1
                    and result.tool_name == "search_campus_knowledge"
                    and result.output.startswith("知识库中暂未找到相关信息")
                ):
                    continue
                sections.append(result.output)
            else:
                sections.append(f"{result.tool_name} 未能完成：{result.error}")
        return "\n\n".join(sections)

    @staticmethod
    def _extract_weekday(text: str) -> str | None:
        lowered = text.lower()
        # 优先匹配更长的别名，防止“星期日”被较短字符串误处理。
        for alias in sorted(WEEKDAY_ALIASES, key=len, reverse=True):
            if alias in lowered:
                return WEEKDAY_ALIASES[alias]
        return None

    @classmethod
    def _extract_service_code(cls, text: str) -> str | None:
        return next(
            (code for alias, code in cls._service_aliases if alias in text),
            None,
        )

    @staticmethod
    def _extract_campus(text: str) -> str:
        if any(alias in text for alias in ("泰山", "东区")):
            return "east"
        if any(alias in text for alias in ("启林", "西区")):
            return "west"
        return "main"

    @classmethod
    def _knowledge_only_query(cls, query: str) -> str:
        """混合问题中去掉由实时业务工具负责的独立分句。"""

        clauses = [
            clause.strip()
            for clause in re.split(r"[，,。；;！？?!]+", query)
            if clause.strip()
        ]
        static_clauses = [
            clause
            for clause in clauses
            if not any(word in clause for word in cls._live_status_words)
        ]
        return "，".join(static_clauses) or query
