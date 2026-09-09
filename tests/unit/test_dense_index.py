"""UNIT TEST -- index construction and the dense channel. No model weights,
no ChromaDB. Fakes stand in for both, so this is NOT evidence that embedding or
Chroma retrieval works; that is a separate offline integration test run against
the real snapshot."""

from __future__ import annotations

import pytest

from georef.retrieval.dense import build_index

RECORDS = [
    {"linz_id": "2", "name": "Akaroa", "feature_type": "Locality", "land_district": "Canterbury",
     "latitude": -43.81, "longitude": 172.97, "source": "LINZ", "dataset_version": "20260908",
     "geometry_type": "POINT", "name_status": "Official Assigned", "country": "New Zealand"},
    {"linz_id": "1", "name": "Waiouru", "feature_type": "Locality", "land_district": "Wellington",
     "latitude": -39.48, "longitude": 175.67, "source": "LINZ", "dataset_version": "20260908",
     "geometry_type": "POINT", "name_status": "Official Assigned", "country": "New Zealand"},
]


class FakeEmbedder:
    model_name = "fake/embedder"

    def __init__(self):
        self.batches = []

    def encode(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class FakeStore:
    collection_name = "nz_geographic_entities"

    def __init__(self, miscount: int | None = None):
        self.ids, self.metadatas, self.documents = [], [], []
        self.resets = 0
        self._miscount = miscount

    def reset(self):
        self.resets += 1
        self.ids, self.metadatas, self.documents = [], [], []

    def add(self, ids, embeddings, documents, metadatas):
        self.ids += list(ids)
        self.documents += list(documents)
        self.metadatas += list(metadatas)

    def count(self):
        return self._miscount if self._miscount is not None else len(self.ids)


def document_for(record):
    return f"{record['name']} | {record['feature_type']} | {record['land_district']} | New Zealand"


class TestBuildIndex:
    def test_indexes_every_record(self):
        store = FakeStore()
        result = build_index(RECORDS, FakeEmbedder(), store, document_for, "snap.jsonl", "abc")
        assert result.record_count == 2
        assert set(store.ids) == {"1", "2"}

    def test_resets_before_loading(self):
        """An incremental add would leave records from a superseded snapshot behind."""
        store = FakeStore()
        build_index(RECORDS, FakeEmbedder(), store, document_for, "snap.jsonl", "abc")
        assert store.resets == 1

    def test_a_partial_index_is_an_error(self):
        store = FakeStore(miscount=1)
        with pytest.raises(RuntimeError, match="silently narrow retrieval"):
            build_index(RECORDS, FakeEmbedder(), store, document_for, "snap.jsonl", "abc")

    def test_every_record_carries_its_authoritative_source(self):
        store = FakeStore()
        build_index(RECORDS, FakeEmbedder(), store, document_for, "snap.jsonl", "abc")
        assert all(m["source"] == "LINZ" for m in store.metadatas)
        assert all(m["dataset_version"] == "20260908" for m in store.metadatas)

    def test_metadata_keeps_coordinates_numeric(self):
        store = FakeStore()
        build_index(RECORDS, FakeEmbedder(), store, document_for, "snap.jsonl", "abc")
        assert isinstance(store.metadatas[0]["latitude"], float)

    def test_records_the_snapshot_it_was_built_from(self):
        result = build_index(RECORDS, FakeEmbedder(), FakeStore(), document_for, "snap.jsonl", "abc")
        assert result.snapshot_file == "snap.jsonl"
        assert result.snapshot_sha256 == "abc"

    def test_embedder_sees_the_same_document_text_as_bm25(self):
        embedder = FakeEmbedder()
        build_index(RECORDS, embedder, FakeStore(), document_for, "snap.jsonl", "abc")
        assert embedder.batches[0][0] == document_for(RECORDS[0])

    def test_batches_cover_the_whole_snapshot(self):
        store = FakeStore()
        many = [{**RECORDS[0], "linz_id": str(i)} for i in range(2500)]
        build_index(many, FakeEmbedder(), store, document_for, "snap.jsonl", "abc", batch_size=1000)
        assert len(store.ids) == 2500
