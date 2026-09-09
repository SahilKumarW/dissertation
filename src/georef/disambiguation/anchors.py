"""Anchor strings, and the forms worth looking up.

The model reports the anchor as the description writes it, which is the right
thing for provenance and the wrong thing for a gazetteer lookup. Real examples
from the dev run:

    "Minaret Creek [Minaret Burn]"   bracketed alternative name
    "upper end of Lake Orbell"       positional qualifier
    "the town of Otautau"            settlement wrapper
    "the bridge across the stream"   no named entity at all

Each is handled by a general rule, never by a list of specific places. The
original string is always preserved and always tried first; variants are
additional query forms, not replacements, so a variant can only add candidates.

Generic-term handling (master spec section 21) is deliberately two-sided. A
generic word is not discarded, because "Lake Rotoiti" and "Porarari River"
carry real feature-type information in exactly those words. What is flagged is
an anchor that is *only* generic terms, which no gazetteer lookup can resolve.
"""

from __future__ import annotations

import re

from georef.normalise import match_key, normalise

__all__ = [
    "GENERIC_TERMS",
    "anchor_variants",
    "is_generic_only",
    "feature_terms",
    "distinctive_tokens",
    "name_similarity",
]

# Generic geographic terms. Used for two different purposes: detecting an anchor
# that names no specific place, and inferring which feature type a candidate
# should be.
GENERIC_TERMS: dict[str, tuple[str, ...]] = {
    "river": ("River", "Stream"),
    "stream": ("Stream", "River"),
    "creek": ("Stream", "River"),
    "burn": ("Stream", "River"),
    "brook": ("Stream",),
    "lake": ("Lake",),
    "tarn": ("Lake", "Tarn"),
    "lagoon": ("Lagoon", "Lake"),
    "bay": ("Bay",),
    "beach": ("Beach",),
    "harbour": ("Harbour",),
    "inlet": ("Inlet", "Bay"),
    "sound": ("Sound",),
    "island": ("Island",),
    "point": ("Point",),
    "head": ("Point", "Head"),
    "cape": ("Cape", "Point"),
    "peninsula": ("Peninsula",),
    "hill": ("Hill",),
    "mount": ("Hill", "Mountain", "Peak"),
    # Abbreviated forms, as they appear on collection labels.
    "mt": ("Hill", "Mountain", "Peak"),
    "mtn": ("Hill", "Mountain", "Peak"),
    "pt": ("Point",),
    "mountain": ("Hill", "Mountain", "Peak"),
    "peak": ("Peak", "Hill"),
    "summit": ("Hill", "Peak", "Mountain"),
    "ridge": ("Ridge",),
    "range": ("Range",),
    "spur": ("Spur",),
    "saddle": ("Saddle", "Pass"),
    "pass": ("Pass",),
    "valley": ("Valley",),
    "gully": ("Gully", "Valley"),
    "gorge": ("Gorge",),
    "flat": ("Flat",),
    "plain": ("Plain",),
    "swamp": ("Swamp", "Wetland"),
    "falls": ("Falls",),
    "glacier": ("Glacier",),
    "forest": ("Forest",),
    "reserve": ("Scenic Reserve", "Recreation Reserve", "Reserve"),
    "park": ("Park", "National Park"),
    "station": ("Railway Station", "Locality"),
    "township": ("Locality", "Suburb"),
    "town": ("Locality", "Suburb"),
    "village": ("Locality", "Suburb"),
    "settlement": ("Locality",),
    "suburb": ("Suburb", "Locality"),
    "trig": ("Trig Station", "Hill"),
    "hut": ("Hut",),
    "bridge": ("Bridge",),
    "road": ("Road",),
    "track": ("Track",),
    "clearing": ("Clearing",),
    "mine": ("Mine",),
}

# Words that describe a position relative to a feature rather than naming one.
_QUALIFIER = re.compile(
    r"^(?:the\s+)?(?:upper|lower|middle|mid|north(?:ern)?|south(?:ern)?|east(?:ern)?|"
    r"west(?:ern)?|top|bottom|head|mouth|end|edge|side|summit|foot|base|outlet|"
    r"junction|jn|confluence)\s+(?:end\s+|reaches\s+|slopes?\s+|part\s+)?of\s+",
    re.IGNORECASE,
)

_SETTLEMENT_WRAPPER = re.compile(
    r"^(?:the\s+)?(?:town|township|village|settlement|city|locality|suburb|port)\s+of\s+",
    re.IGNORECASE,
)

_TRAILING_DESCRIPTOR = re.compile(
    r"\s+(?:township|village|settlement|town|area|district|region|mouth|bridge)$",
    re.IGNORECASE,
)

_BRACKETED = re.compile(r"\s*[\[\(]([^\]\)]+)[\]\)]")

_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def anchor_variants(anchor: str | None) -> list[str]:
    """Query forms for an anchor, most literal first, without duplicates.

    The original is always first. Everything after it is a strictly additional
    query, so a variant can only widen the candidate pool, never replace a match
    the original would have found.
    """
    if not anchor or not anchor.strip():
        return []

    original = anchor.strip()
    variants: list[str] = [original]

    def add(candidate: str) -> None:
        candidate = candidate.strip(" ,.;:-")
        if candidate and not any(match_key(candidate) == match_key(v) for v in variants):
            variants.append(candidate)

    # "Minaret Creek [Minaret Burn]" -> both names, separately.
    bracketed = _BRACKETED.search(original)
    if bracketed:
        add(_BRACKETED.sub("", original))
        add(bracketed.group(1))

    working = _BRACKETED.sub("", original).strip()

    # "the town of Otautau" -> "Otautau"
    unwrapped = _SETTLEMENT_WRAPPER.sub("", working)
    if unwrapped != working:
        add(unwrapped)
        working = unwrapped

    # "upper end of Lake Orbell" -> "Lake Orbell"
    dequalified = _QUALIFIER.sub("", working)
    if dequalified != working:
        add(dequalified)
        working = dequalified

    # "Erua township" -> "Erua"
    trimmed = _TRAILING_DESCRIPTOR.sub("", working)
    if trimmed != working:
        add(trimmed)

    without_article = _LEADING_ARTICLE.sub("", working)
    if without_article != working:
        add(without_article)

    return variants


def feature_terms(anchor: str | None) -> tuple[str, ...]:
    """Feature types implied by generic words inside the anchor.

    "Porarari River" implies a Stream or River candidate; "Lake Camp" implies a
    Lake. General rules only -- no specific place is ever named here.
    """
    if not anchor:
        return ()
    tokens = re.findall(r"[a-z]+", match_key(anchor))
    types: list[str] = []
    for token in tokens:
        for feature_type in GENERIC_TERMS.get(token, ()):
            if feature_type not in types:
                types.append(feature_type)
    return tuple(types)


def is_generic_only(anchor: str | None) -> bool:
    """True when the anchor names no specific place.

    "the bridge across the stream" and "the summit" cannot be resolved against a
    gazetteer of proper names, however good retrieval is. Detecting that at
    prediction time turns a guaranteed wrong answer into an honest abstention.
    """
    if not anchor:
        return True
    tokens = re.findall(r"[a-z]+", normalise(anchor))
    stopwords = {"the", "a", "an", "of", "at", "on", "in", "near", "by", "across", "to", "and"}
    meaningful = [token for token in tokens if token not in stopwords]
    if not meaningful:
        return True
    return all(token in GENERIC_TERMS for token in meaningful)


_STOPWORDS = {"the", "a", "an", "of", "at", "on", "in", "near", "by", "across", "to", "and"}


def distinctive_tokens(text: str | None) -> set[str]:
    """Tokens that identify WHICH place, with generic terms removed.

    "Mount Benoit" and "Mount Guinevere" share "mount", which says what kind of
    feature it is and nothing about which one. Comparing names on the remaining
    tokens is what separates a partial match from a coincidence.
    """
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9]+", match_key(text))
    return {t for t in tokens if t not in _STOPWORDS and t not in GENERIC_TERMS}


def name_similarity(anchor: str | None, candidate_name: str | None) -> float:
    """Overlap of distinctive tokens, in [0, 1].

    Containment scores higher than plain overlap, because "Te Pukaha" is a
    plausible gazetteer form of the anchor "Te Pukaha Trig" -- the extra token
    is a feature type, not a different place. Two names sharing only a generic
    word score zero.
    """
    left = distinctive_tokens(anchor)
    right = distinctive_tokens(candidate_name)
    if not left or not right:
        return 0.0
    shared = left & right
    if not shared:
        return 0.0
    if left == right:
        # "Te Pukaha Trig" and "Te Pukaha" differ only by a generic word, so
        # after removing generics they name the same thing.
        return 1.0
    if left <= right or right <= left:
        # One name adds a distinctive word the other lacks: "Mount George" and
        # "Mount Saint George" are related but not the same place.
        return 0.7
    return len(shared) / len(left | right)


# ---------------------------------------------------------------------------
# Back-reference resolution
# ---------------------------------------------------------------------------
#
# Descriptions name a feature once and then refer back to it:
#
#     "Lindis Pass, Stream gully crossing Highway 8 about 200m west of Pass"
#     "Mt. Maungatua south west of Dunedin, rock c. 500 m west of the summit"
#     "S.W. Nelson, Perpendicular Point by trig. 1/2 mile N of the point"
#
# The referent is present in the description, so resolving it is text
# processing, not geography. This runs on the description string alone: it
# never consults the gazetteer, so it cannot invent a place, and the resolved
# anchor still has to survive retrieval and disambiguation like any other.
#
# Deliberately conservative. It resolves only when the description contains a
# named phrase that shares the anchor's generic word or its implied feature
# type, and returns None otherwise -- "100m west of hut track" names nothing and
# stays an abstention.

_NAMED_PHRASE = re.compile(
    r"\b(?:(?:Mt|Mount|Lake|Cape|Port|St)\.?\s+)?"
    r"[A-Z][A-Za-z\u0100-\u017f'\u2019-]+"
    r"(?:\s+[A-Z][A-Za-z\u0100-\u017f'\u2019-]+){0,3}"
)


def named_phrases(description: str | None) -> list[str]:
    """Capitalised multi-word phrases in a description, longest first.

    A crude proper-noun finder. It over-generates -- sentence-initial words and
    collector names come through too -- which is safe here because a phrase only
    matters if it also matches the anchor's feature type, and any survivor still
    has to be found in the gazetteer.
    """
    if not description:
        return []
    seen: list[str] = []
    for match in _NAMED_PHRASE.finditer(description):
        phrase = match.group(0).strip(" .,;:")
        if phrase and phrase not in seen and distinctive_tokens(phrase):
            seen.append(phrase)
    return sorted(seen, key=len, reverse=True)


def resolve_back_reference(anchor: str | None, description: str | None) -> str | None:
    """Resolve a generic anchor to the feature the description names.

    Returns None when the description offers no match, so the caller abstains
    rather than substituting something arbitrary.
    """
    if not anchor or not description or not is_generic_only(anchor):
        return None

    anchor_generics = [
        token for token in re.findall(r"[a-z]+", normalise(anchor)) if token in GENERIC_TERMS
    ]
    if not anchor_generics:
        return None

    phrases = named_phrases(description)

    # First choice: a phrase ending in the same generic word.
    # "Pass" -> "Lindis Pass";  "the point" -> "Perpendicular Point"
    for generic in anchor_generics:
        for phrase in phrases:
            tokens = re.findall(r"[a-z]+", normalise(phrase))
            if len(tokens) > 1 and tokens[-1] == generic:
                return phrase

    # Second choice: a phrase whose implied feature type matches the anchor's.
    # "the summit" implies Hill/Peak/Mountain; "Mt. Maungatua" implies the same.
    expected = set(feature_terms(anchor))
    if expected:
        for phrase in phrases:
            if expected & set(feature_terms(phrase)):
                return phrase

    return None
