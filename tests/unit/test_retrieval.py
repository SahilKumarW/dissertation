"""UNIT TEST -- hybrid retrieval. Real BM25, fake embedder, fake reranker.

The lexical channels and fusion run for real against a small synthetic
gazetteer. The dense channel and reranker are stubs, so this is NOT evidence
that embedding or cross-encoder reranking works end to end; that needs the
model weights and is a separate offline integration test.
"""

from __future__ import annotations

import pytest

from georef.config.loader import load_yaml, repo_root
from georef.retrieval.candidates import reciprocal_rank_fusion
from georef.retrieval.hybrid import HybridRetriever
from georef.retrieval.lexical import LexicalIndex, tokenise

CONFIG = load_yaml(repo_root() / "configs" / "retrieval.yaml")

GAZETTEER = [
    {"linz_id": "1", "name": "Waiouru", "feature_type": "Locality",
     "land_district": "Wellington", "latitude": -39.4783, "longitude": 175.6689, "source": "LINZ"},
    {"linz_id": "2", "name": "Waiouru", "feature_type": "Railway Station",
     "land_district": "Wellington", "latitude": -39.4800, "longitude": 175.6700, "source": "LINZ"},
    {"linz_id": "3", "name": "\u014cp\u014dtiki", "feature_type": "Locality",
     "land_district": "Gisborne", "latitude": -38.0055, "longitude": 177.2870, "source": "LINZ"},
    {"linz_id": "4", "name": "Akaroa", "feature_type": "Locality",
     "land_district": "Canterbury", "latitude": -43.8103, "longitude": 172.9679, "source": "LINZ"},
    {"linz_id": "5", "name": "Akaroa Harbour", "feature_type": "Harbour",
     "land_district": "Canterbury", "latitude": -43.8500, "longitude": 172.9500, "source": "LINZ"},
    {"linz_id": "6", "name": "Porarari River", "feature_type": "Stream",
     "land_district": "Nelson", "latitude": -42.1100, "longitude": 171.3400, "source": "LINZ"},
]


@pytest.fixture(scope="module")
def index() -> LexicalIndex:
    return LexicalIndex(GAZETTEER, CONFIG["document_template"])


class FakeEmbedder:
    """Deterministic bag-of-characters vectors; no model weights involved."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 26
            for character in text.lower():
                if "a" <= character <= "z":
                    vector[ord(character) - 97] += 1.0
            vectors.append(vector)
        return vectors


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def query(self, embedding, top_k):
        self.queries.append((embedding, top_k))
        return self.results[:top_k]


class FakeReranker:
    """Scores by a caller-supplied mapping so ordering effects are visible."""

    def __init__(self, scores_by_name):
        self.scores_by_name = scores_by_name
        self.seen = []

    def score(self, query, documents):
        self.seen.append((query, list(documents)))
        return [
            next((v for k, v in self.scores_by_name.items() if k in document), 0.0)
            for document in documents
        ]


class TestTokenisation:
    def test_folds_macrons(self):
        assert tokenise("\u014cp\u014dtiki") == tokenise("Opotiki")

    def test_splits_on_punctuation_and_case(self):
        assert tokenise("Akaroa Harbour | Harbour") == ["akaroa", "harbour", "harbour"]


class TestLexicalChannels:
    def test_exact_returns_every_record_sharing_a_name(self, index):
        """41,402 distinct names across 49,771 records: a name is not an answer."""
        assert index.exact_name("Waiouru") == ["1", "2"]

    def test_exact_is_case_and_punctuation_insensitive(self, index):
        assert index.exact_name("  waiouru. ") == ["1", "2"]

    def test_exact_misses_macronised_names(self, index):
        assert index.exact_name("Opotiki") == []

    def test_folded_channel_recovers_them(self, index):
        assert index.folded_name("Opotiki") == ["3"]

    def test_bm25_finds_partial_names(self, index):
        identifiers, scores = index.bm25("Akaroa Harbour", top_k=5)
        assert "5" in identifiers
        assert scores["5"] > 0

    def test_bm25_uses_feature_type_evidence(self, index):
        identifiers, _ = index.bm25("Porarari River stream", top_k=5)
        assert identifiers[0] == "6"

    def test_empty_query_returns_nothing(self, index):
        assert index.bm25("", top_k=5) == ([], {})


class TestFusion:
    def test_rewards_agreement_across_channels(self):
        fused = reciprocal_rank_fusion({"a": ["x", "y"], "b": ["y", "z"]}, k=60)
        assert fused[0][0] == "y"  # ranked by both channels

    def test_records_each_channel_rank(self):
        fused = reciprocal_rank_fusion({"a": ["x"], "b": ["x"]}, k=60)
        assert fused[0][2] == {"a": 1, "b": 1}

    def test_ties_break_deterministically(self):
        first = reciprocal_rank_fusion({"a": ["b1", "a1"]}, k=60)
        second = reciprocal_rank_fusion({"a": ["b1", "a1"]}, k=60)
        assert [i for i, _, _ in first] == [i for i, _, _ in second]

    def test_equal_scores_order_by_identifier(self):
        fused = reciprocal_rank_fusion({"a": ["z"], "b": ["y"]}, k=60)
        assert [i for i, _, _ in fused] == ["y", "z"]

    def test_rejects_an_unknown_tie_break_policy(self):
        with pytest.raises(ValueError, match="tie_break"):
            reciprocal_rank_fusion({"a": ["x"]}, tie_break="whatever")


class TestHybridRetriever:
    def test_fuses_lexical_channels_without_a_vector_store(self, index):
        retriever = HybridRetriever(index, CONFIG)
        candidates = retriever.retrieve("Waiouru")
        assert {c.linz_id for c in candidates} >= {"1", "2"}
        assert candidates[0].fused_score > 0

    def test_macronised_record_is_reachable_from_an_unmacronised_query(self, index):
        retriever = HybridRetriever(index, CONFIG)
        assert "3" in {c.linz_id for c in retriever.retrieve("Opotiki")}

    def test_dense_channel_contributes_when_supplied(self, index):
        store = FakeVectorStore([("5", 0.91), ("4", 0.72)])
        retriever = HybridRetriever(index, CONFIG, FakeEmbedder(), store)
        candidates = retriever.retrieve("Akaroa")
        assert store.queries, "dense channel was not consulted"
        by_id = {c.linz_id: c for c in candidates}
        assert by_id["5"].channel_scores["dense"] == 0.91

    def test_candidates_carry_their_channel_provenance(self, index):
        retriever = HybridRetriever(index, CONFIG)
        candidate = retriever.retrieve("Waiouru")[0]
        provenance = candidate.to_provenance()
        assert provenance["source"] == "LINZ"
        assert provenance["channel_ranks"]
        assert provenance["linz_id"] in {"1", "2"}

    def test_unknown_query_produces_no_exact_evidence(self, index):
        """BM25 always returns its top k, so absence shows up as missing exact
        evidence rather than an empty list. The disambiguator, not retrieval,
        decides that nothing is good enough."""
        candidates = HybridRetriever(index, CONFIG).retrieve("Nowhereville")
        assert all("exact_name" not in c.channel_ranks for c in candidates)
        assert all("folded_name" not in c.channel_ranks for c in candidates)

    def test_reranker_reorders_and_records_scores(self, index):
        reranker = FakeReranker({"Railway Station": 9.0, "Locality": 1.0})
        retriever = HybridRetriever(index, CONFIG, reranker=reranker)
        candidates = retriever.retrieve("Waiouru railway")
        assert candidates[0].linz_id == "2"
        assert candidates[0].rerank_score == 9.0

    def test_reranker_sees_the_same_document_text_as_bm25(self, index):
        """Both channels must judge the same evidence, or their scores are not
        about the same thing."""
        reranker = FakeReranker({})
        retriever = HybridRetriever(index, CONFIG, reranker=reranker)
        retriever.retrieve("Waiouru")
        query, documents = reranker.seen[0]
        assert query == "Waiouru"
        expected = {index.document_text(record) for record in GAZETTEER}
        assert set(documents) <= expected
        assert "Waiouru | Locality | Wellington | New Zealand" in documents

    def test_respects_final_top_k(self, index):
        config = {**CONFIG, "final_top_k": 2, "rerank": {**CONFIG["rerank"], "enabled": False}}
        assert len(HybridRetriever(index, config).retrieve("Akaroa")) <= 2
