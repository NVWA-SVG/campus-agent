import unittest

from campus_agent.cli import build_parser


class CliTests(unittest.TestCase):
    def test_default_planner_is_offline_rule(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.planner, "rule")

    def test_deepseek_planner_must_be_selected_explicitly(self) -> None:
        args = build_parser().parse_args(["--planner", "deepseek"])
        self.assertEqual(args.planner, "deepseek")


if __name__ == "__main__":
    unittest.main()
