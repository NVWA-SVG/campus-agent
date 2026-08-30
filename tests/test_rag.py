from __future__ import annotations

import unittest

from campus_agent.rag import LocalRAG
from campus_agent.rag.chunking import load_markdown_text
from campus_agent.rag.retriever import tokenize


class RAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rag = LocalRAG()

    def test_markdown_is_split_into_sections(self) -> None:
        self.assertEqual(len(self.rag.chunks), 10)
        self.assertTrue(
            all(chunk.source.endswith(chunk.title) for chunk in self.rag.chunks)
        )

    def test_multiple_level_one_headings_do_not_merge_sections(self) -> None:
        chunks = load_markdown_text(
            "# 第一份资料\n第一段内容\n# 第二份资料\n第二段内容",
            source_name="combined.md",
            chunk_id_prefix="combined",
            default_title="combined",
        )
        self.assertEqual([chunk.title for chunk in chunks], ["第一份资料", "第二份资料"])
        self.assertEqual([chunk.content for chunk in chunks], ["第一段内容", "第二段内容"])

    def test_tokenizer_supports_chinese_bigrams_and_english(self) -> None:
        tokens = tokenize("校园卡 API")
        self.assertIn("校园", tokens)
        self.assertIn("园卡", tokens)
        self.assertIn("api", tokens)

    def test_retrieves_campus_card_process(self) -> None:
        hits = self.rag.retrieve("饭卡丢了怎么挂失补办？")
        self.assertTrue(hits)
        self.assertIn("campus_card.md", hits[0].chunk.source)
        self.assertIn("挂失与补办", hits[0].chunk.title)

    def test_retrieves_laboratory_incident_process(self) -> None:
        hits = self.rag.retrieve("实验室设备坏了应该如何处理？")
        self.assertTrue(hits)
        self.assertIn("取消与异常", hits[0].chunk.title)

    def test_answer_contains_source_citation(self) -> None:
        answer = self.rag.answer("怎样申请英文成绩单？")
        self.assertIn("来源：", answer.answer)
        self.assertIn("transcript.md", answer.answer)

    def test_unmatched_query_returns_conservative_answer(self) -> None:
        answer = self.rag.answer("火星基地如何办理通行证？")
        self.assertEqual(answer.hits, ())
        self.assertIn("暂未找到", answer.answer)

    def test_related_topic_without_requested_price_abstains(self) -> None:
        answer = self.rag.answer("校园卡补办多少钱？")

        self.assertTrue(answer.hits)
        self.assertIn("没有提供", answer.answer)
        self.assertIn("费用或金额", answer.answer)
        self.assertNotIn("20元", answer.answer)

    def test_all_precise_missing_fact_forms_abstain(self) -> None:
        queries = (
            "校园卡挂失后余额冻结要等待几分钟才生效？",
            "借来的图书丢失后具体按书价几倍赔偿？",
            "一次最多能申请多少份中文成绩单？",
            "实验室最早可以提前多少天预约？",
        )

        for query in queries:
            with self.subTest(query=query):
                answer = self.rag.answer(query)
                self.assertTrue(answer.hits)
                self.assertIn("没有提供", answer.answer)

    def test_precise_fact_present_in_best_evidence_is_answered(self) -> None:
        chunks = load_markdown_text(
            "# 自助打印指南\n\n## 收费标准\n\n自助打印每份收费2元。",
            source_name="printing.md",
            chunk_id_prefix="printing",
            default_title="自助打印指南",
        )
        rag = LocalRAG(extra_chunk_loader=lambda: chunks)

        answer = rag.answer("自助打印每份收费多少钱？")

        self.assertTrue(answer.hits)
        self.assertEqual(answer.hits[0].chunk.chunk_id, "printing-1-0")
        self.assertNotIn("没有提供", answer.answer)
        self.assertIn("2元", answer.answer)

    def test_unrelated_numeric_hit_cannot_satisfy_missing_card_price(self) -> None:
        chunks = load_markdown_text(
            "# 自助打印指南\n\n## 收费标准\n\n自助打印每份收费2元。",
            source_name="printing.md",
            chunk_id_prefix="printing",
            default_title="自助打印指南",
        )
        rag = LocalRAG(extra_chunk_loader=lambda: chunks)

        answer = rag.answer("校园卡补办具体收费多少元？")

        self.assertEqual(answer.hits[0].chunk.chunk_id, "campus_card-1-0")
        self.assertIn("没有提供", answer.answer)

    def test_normal_process_question_is_not_mistaken_for_missing_fact(self) -> None:
        answer = self.rag.answer("校园卡丢失后如何挂失和补办？")

        self.assertTrue(answer.hits)
        self.assertNotIn("没有提供", answer.answer)
        self.assertIn("挂失", answer.answer)


if __name__ == "__main__":
    unittest.main()
