"""Spotting gazetteer names inside a description, before interpretation.

This is the "query first" half of a two-pass design: look the description up in
the local gazetteer, tell the model which of the names it contains are real New
Zealand places, then interpret. The second pass is the usual grounding query
after interpretation.

**Names only. Never coordinates.** The responsibility split in the architecture
holds precisely because the model does not know where anything is. Handing it a
list of names it can see in front of it does not change that: it still cannot
invent an authoritative coordinate, still does not choose between two places of
the same name, and still does not compute anything. Handing it coordinates
would change it, so this module never returns any.

The spotting itself is deterministic: n-grams of the description are looked up
against name keys built from the frozen snapshot. Same description, same
snapshot, same hints, every time.
"""

from __future__ import annotations

import re
from collections import defaultdict

from georef.disambiguation.anchors import distinctive_tokens
from georef.normalise import match_key

__all__ = ["NameSpotter"]

# Only phrases that look like proper nouns in the source text are considered.
# Without this, common words that happen to be gazetteer names ("Bush", "Flat",
# "The Sisters") match constantly and bury the useful hints.
_TOKEN = re.compile(r"[A-Za-z\u0100-\u017f'\u2019]+")


class NameSpotter:
    """Finds gazetteer names occurring in a description."""

    def __init__(
        self,
        records: list[dict],
        max_name_tokens: int = 4,
        ambiguity_radius_km: float = 25.0,
    ):
        self.max_name_tokens = max_name_tokens
        # A context name is only evidence about WHERE the description is if it
        # identifies one place. "North Island" and "Bay" occur all over the
        # gazetteer, and a scattered name used as a reference point makes any
        # candidate look coherent with something.
        self.ambiguity_radius_km = ambiguity_radius_km
        # key -> the canonical spellings that share it, so a hint can be offered
        # in the gazetteer's own form rather than the description's.
        self._by_key: dict[str, list[str]] = defaultdict(list)
        self._records_by_key: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            name = record.get("name")
            if not name:
                continue
            if not distinctive_tokens(name):
                # A name made only of generic words ("The Pass", "Township")
                # cannot identify a place and would fire everywhere.
                continue
            for key in self._keys_for(name):
                if name not in self._by_key[key]:
                    self._by_key[key].append(name)
                self._records_by_key[key].append(record)

    @staticmethod
    def _keys_for(name: str) -> set[str]:
        """Every key a gazetteer name should be findable under.

        Two forms matter in this gazetteer:

        * Dual names. LINZ gazettes "Wairewa/Lake Forsyth" and "Cook
          River/Weheka" as single names, so a description writing only one side
          matches nothing without splitting on the slash.
        * Spacing. A description writes "Lake Clear Water" where the gazetteer
          has "Lake Clearwater", so a space-free key is kept as a fallback.
        """
        keys = set()
        parts = [name] + [part for part in name.split("/") if part.strip()]
        for part in parts:
            key = match_key(part)
            if not key:
                continue
            keys.add(key)
            keys.add(key.replace(" ", ""))
        return keys

    def spot(self, description: str | None, limit: int = 12) -> list[str]:
        """Gazetteer names appearing in the description, longest match first.

        Longer matches are preferred and shorter matches contained within them
        are dropped: "Lindis Pass" is reported, "Lindis" alone is not, because
        the longer form is the one the description actually uses.
        """
        if not description:
            return []

        tokens = [
            (match.group(0), match.start(), match.end())
            for match in _TOKEN.finditer(description)
        ]
        if not tokens:
            return []

        found: list[tuple[int, int, str]] = []
        for size in range(min(self.max_name_tokens, len(tokens)), 0, -1):
            for start in range(len(tokens) - size + 1):
                window = tokens[start : start + size]
                # Proper-noun shape: every word capitalised in the source text.
                if not all(word[0].isupper() for word, _, _ in window):
                    continue
                phrase = description[window[0][1] : window[-1][2]]
                # "Mt. Maungatua" carries an internal period the gazetteer form
                # lacks. Tried as a second key here rather than by changing
                # match_key, which the disambiguator also depends on.
                names = None
                for key in self._keys_for(phrase) | self._keys_for(phrase.replace(".", " ")):
                    names = self._by_key.get(key)
                    if names:
                        break
                if not names:
                    continue
                span = (window[0][1], window[-1][2])
                if any(span[0] >= s and span[1] <= e for s, e, _ in found):
                    continue  # already covered by a longer match
                found.append((span[0], span[1], names[0]))

        ordered: list[str] = []
        for _, _, name in sorted(found):
            if name not in ordered:
                ordered.append(name)
        return ordered[:limit]


    def _is_unambiguous(self, records: list[dict]) -> bool:
        """True when every record sharing this name sits in the same locality.

        A name held by features hundreds of kilometres apart locates nothing.
        Extent is measured on a plain lat/lon box rather than a geodesic: this
        is a coarse ambiguity filter, and a degree of latitude is close enough
        to 111 km everywhere in New Zealand for the purpose.
        """
        points = []
        for record in records:
            try:
                points.append((float(record["latitude"]), float(record["longitude"])))
            except (KeyError, TypeError, ValueError):
                continue
        if not points:
            return False
        if len(points) == 1:
            return True
        latitudes = [p[0] for p in points]
        longitudes = [p[1] for p in points]
        span_km = max(
            (max(latitudes) - min(latitudes)) * 111.0,
            (max(longitudes) - min(longitudes)) * 111.0 * 0.75,  # cos(~41 deg S)
        )
        return span_km <= self.ambiguity_radius_km

    def spot_records(self, description: str | None, limit: int = 12) -> list[dict]:
        """The same matches, as full gazetteer records, ambiguous names removed.

        Coordinates ARE included here, deliberately, and this method is for the
        deterministic disambiguator only -- never for building a model prompt.
        The rule that keeps the architecture honest is that the MODEL does not
        learn where anything is; Python already knows, because the snapshot is
        its own authoritative source.

        Names that occur in several distant places are dropped entirely. On the
        dev set, keeping them was worse than having no context at all: "North
        Island" in a description pulled a Puketi in Otago ahead of the Puketi in
        the Bay of Islands, 1,249 km wrong.
        """
        records: list[dict] = []
        for name in self.spot(description, limit):
            matches: list[dict] = []
            seen_ids = set()
            for key in self._keys_for(name):
                for record in self._records_by_key.get(key, []):
                    identifier = id(record)
                    if identifier not in seen_ids:
                        seen_ids.add(identifier)
                        matches.append(record)
            if not self._is_unambiguous(matches):
                continue
            records.extend(matches)
        return records
