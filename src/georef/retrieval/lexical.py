"""Lexical retrieval channels.

Three of the four channels, all in memory and all deterministic:

* ``exact_name``  -- normalise() key lookup
* ``folded_name`` -- match_key() lookup, diacritic-insensitive
* ``bm25``        -- Okapi BM25 over name, feature type and district

Anchors in this corpus are proper nouns, so lexical matching carries most of
the retrieval load; the dense channel exists to catch partial and reworded
forms that exact keys miss.

A name is not unique in this gazetteer -- 49,771 records share 41,402 distinct
names -- so every lookup returns a LIST. A channel that returned one record per
name would be quietly making the disambiguator's decision for it.
"""

from __future__ import annotations

import re
from collections import defaultdict

from georef.normalise import match_key, normalise

__all__ = ["LexicalIndex", "tokenise"]

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Lowercase, diacritic-folded alphanumeric tokens.

    Folding here means 'Ōpōtiki' and 'Opotiki' produce identical tokens, so the
    BM25 channel is macron-insensitive without a separate index.
    """
    return _TOKEN.findall(match_key(text))


class LexicalIndex:
    """In-memory exact, folded and BM25 indexes over the gazetteer snapshot."""

    def __init__(self, records: list[dict], document_template: str):
        self.records: dict[str, dict] = {}
        self.document_template = document_template
        self._exact: dict[str, list[str]] = defaultdict(list)
        self._folded: dict[str, list[str]] = defaultdict(list)
        self._corpus_ids: list[str] = []
        self._bm25 = None

        documents: list[list[str]] = []
        for record in records:
            identifier = str(record["linz_id"])
            self.records[identifier] = record
            self._exact[normalise(record["name"])].append(identifier)
            self._folded[match_key(record["name"])].append(identifier)
            self._corpus_ids.append(identifier)
            documents.append(tokenise(self.document_text(record)))

        # Insertion order follows the snapshot, which is sorted by linz_id, so
        # index construction is reproducible.
        self._build_bm25(documents)

    def document_text(self, record: dict) -> str:
        """The text both BM25 and the embedder see, so the channels agree on evidence."""
        return self.document_template.format(
            name=record.get("name") or "",
            feature_type=record.get("feature_type") or "",
            land_district=record.get("land_district") or "",
        )

    def _build_bm25(self, documents: list[list[str]]) -> None:
        import bm25s

        self._bm25 = bm25s.BM25()
        self._bm25.index(documents, show_progress=False)

    # -- channels ------------------------------------------------------------

    def exact_name(self, query: str, top_k: int = 10) -> list[str]:
        return sorted(self._exact.get(normalise(query), []))[:top_k]

    def folded_name(self, query: str, top_k: int = 10) -> list[str]:
        """Matches ignoring macrons. Returns folded hits that exact matching missed too."""
        return sorted(self._folded.get(match_key(query), []))[:top_k]

    def bm25(self, query: str, top_k: int = 50) -> tuple[list[str], dict[str, float]]:
        tokens = tokenise(query)
        if not tokens or self._bm25 is None:
            return [], {}
        count = min(top_k, len(self._corpus_ids))
        indexes, scores = self._bm25.retrieve([tokens], k=count, show_progress=False)
        identifiers: list[str] = []
        by_identifier: dict[str, float] = {}
        for position in range(indexes.shape[1]):
            identifier = self._corpus_ids[int(indexes[0, position])]
            identifiers.append(identifier)
            by_identifier[identifier] = float(scores[0, position])
        return identifiers, by_identifier

    def __len__(self) -> int:
        return len(self.records)
