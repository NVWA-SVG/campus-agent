"""Campus Agent 命令行入口。"""

from __future__ import annotations

import argparse
import json
import os

from campus_agent.agent import build_default_agent
from campus_agent.deepseek_planner import DeepSeekPlanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校园事务 Agent")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="显示规划和工具调用轨迹",
    )
    parser.add_argument(
        "--planner",
        choices=("rule", "deepseek"),
        default="rule",
        help="规划器：rule完全离线；deepseek仅调用DeepSeek官方API并支持失败回退",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    planner = DeepSeekPlanner.from_environment() if args.planner == "deepseek" else None
    agent = build_default_agent(planner=planner)
    session_id = "cli"

    if args.planner == "deepseek" and os.getenv("DEEPSEEK_API_KEY"):
        network_notice = "将请求DeepSeek官方API；失败时回退规则规划"
    elif args.planner == "deepseek":
        network_notice = "未检测到API Key，将完全离线回退规则规划"
    else:
        network_notice = "完全离线，不发起网络请求"
    print(f"Campus Agent 已启动。规划器={args.planner}；{network_notice}。")
    print("输入 /quit 退出，/history 查看记忆，/clear 清空记忆，/metrics 查看指标。")
    while True:
        try:
            query = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return

        if query in {"/quit", "/exit"}:
            print("再见！")
            return
        if query == "/history":
            history = agent.history(session_id)
            if not history:
                print("当前没有会话记录。")
            for message in history:
                print(f"[{message.role}] {message.content}")
            continue
        if query == "/clear":
            agent.clear_history(session_id)
            print("会话记忆已清空。")
            continue
        if query == "/metrics":
            print(json.dumps(agent.planner_metrics(), ensure_ascii=False, indent=2))
            continue

        response = agent.ask(query, session_id=session_id)
        print(f"Agent：{response.answer}")
        if args.trace:
            print("轨迹：")
            for event in response.events:
                print(f"- [{event.event_type}] {event.detail}")


if __name__ == "__main__":
    main()
