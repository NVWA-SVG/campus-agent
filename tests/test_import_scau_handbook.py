from types import SimpleNamespace
from unittest import TestCase

from campus_agent.rag.chunking import load_markdown_text
from scripts.import_scau_handbook import MAX_CHARS, clean_page, convert


class FakeReader:
    def __init__(self, texts):
        self.pages = [SimpleNamespace(extract_text=lambda text=text: text) for text in texts]
        self.outline = [{"/Title": "本科学生学籍管理实施细则", "index": 1}]

    def get_destination_page_number(self, entry):
        return entry["index"]


class ScauHandbookConversionTests(TestCase):
    def test_long_pdf_keeps_body_page_provenance_and_bounds_chunk_lengths(self):
        reader = FakeReader(["封面和目录"] + [f"{i}\n第十条 缓考办理规定。" + "完整办理条件。" * 70 for i in range(1, 105)])
        markdown, stats = convert(reader, "2026")
        chunks = load_markdown_text(markdown, source_name="手册.md", chunk_id_prefix="handbook", default_title="手册")
        self.assertNotIn("封面和目录", markdown)
        self.assertIn("PDF第105页·印刷第104页", markdown)
        self.assertEqual(stats["pdf_pages"], 105)
        self.assertEqual(stats["markdown_sections"], len(chunks))
        self.assertTrue(all(len(chunk.content) <= MAX_CHARS for chunk in chunks))
        self.assertTrue(all("本科学生学籍管理实施细则" in chunk.title for chunk in chunks))

    def test_empty_body_page_requires_review_instead_of_silent_loss(self):
        with self.assertRaisesRegex(ValueError, "无法提取文本"):
            convert(FakeReader(["封面", ""]), "2026")

    def test_wrapped_policy_bookmarks_are_joined_and_carried_to_next_page(self):
        reader = FakeReader(["封面", "两份规章的共同起始页", "后一份规章的后续正文"])
        reader.outline.append({"/Title": "（2026年修订）", "index": 1})
        markdown, _ = convert(reader, "2026")
        self.assertIn("本科学生学籍管理实施细则（2026年修订）｜PDF第2页", markdown)
        self.assertIn("## 本科学生学籍管理实施细则（2026年修订）｜PDF第3页", markdown)

    def test_page_number_removal_preserves_trailing_numeric_table_value(self):
        text, printed_page = clean_page("52\n第一条 学分\n计算规则\n100")
        self.assertEqual(printed_page, "52")
        self.assertIn("学分计算规则100", text)
