"""Contract tests for dataset separation and the RAG retrieval boundary."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import validate
from rag import Retriever


class Contracts(unittest.TestCase):
    def test_disjoint_splits(self):
        counts = validate()
        self.assertEqual(counts["train"], 2592)
        self.assertEqual(counts["validation"], 288)
        self.assertEqual(counts["test"], 576)

    def test_knowledge_update_without_training(self):
        docs = [{"id": "shipping", "title": "Shipping cost", "text": "Shipping cost is AED 15."}]
        self.assertIn("AED 15", Retriever(docs).search("shipping cost")[0]["text"])
        docs[0]["text"] = "Shipping cost is AED 20."
        self.assertIn("AED 20", Retriever(docs).search("shipping cost")[0]["text"])

    def test_no_match_is_empty(self):
        self.assertEqual(Retriever().search("zxqv"), [])

    def test_retrieval_returns_evidence_not_generated_answer(self):
        hits = Retriever().search("minimum spend free shipping")
        self.assertEqual(hits[0]["id"], "shipping-standard")
        self.assertIn("AED 150", hits[0]["text"])


if __name__ == "__main__":
    unittest.main()
