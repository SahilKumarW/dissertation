"""Deterministic text normalisation.

This module is shared by the benchmark builder and, later, by the prediction
pipeline, so that a description is keyed identically wherever it is handled.

Scope (master spec section 12): whitespace, Unicode forms, apostrophes and
quotation marks, case, and trailing punctuation.

Deliberately OUT of scope at this stage: expansion of geographic abbreviations
("Mt" -> "Mount", "Co." -> "County"). Those transformations change meaning-
bearing tokens and belong to the prediction-time normaliser, where they can be
tested against interpretation behaviour. Introducing them here would silently
alter which source rows a benchmark description joins to.

The original string is never mutated in place; callers keep both forms.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalise", "normalisation_version"]

# Version string recorded in provenance. Bump whenever the behaviour of
# normalise() changes, because the benchmark join key depends on it.
NORMALISATION_VERSION = "normalise-v1"

_APOSTROPHES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u00b4": "'",
    "`": "'",
}

_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2033": '"',
}

_DASHES = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
}

_TRANSLATION = str.maketrans({**_APOSTROPHES, **_QUOTES, **_DASHES})

_WHITESPACE = re.compile(r"\s+")
_TRAILING = " \t\n\r.,;:-"


def normalise(text: object) -> str:
    """Return the canonical form of ``text`` used as a join and lookup key.

    Steps, in fixed order:

    1. coerce to ``str`` and apply Unicode NFKC compatibility composition;
    2. map curly apostrophes, quotation marks and dash variants to ASCII;
    3. replace non-breaking and other exotic spaces with a plain space;
    4. collapse runs of whitespace to a single space and strip the ends;
    5. lowercase;
    6. strip trailing punctuation that carries no geographic meaning.

    ``None`` and NaN-like inputs return an empty string rather than the literal
    text "nan", so that missing values cannot accidentally join to each other.
    """
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN
        return ""

    s = unicodedata.normalize("NFKC", str(text))
    s = s.translate(_TRANSLATION)
    # NFKC leaves a few space characters alone; normalise them explicitly.
    s = s.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
    s = _WHITESPACE.sub(" ", s).strip()
    s = s.lower()
    s = s.strip(_TRAILING)
    return s


def normalisation_version() -> str:
    """Version identifier for the normalisation behaviour, for provenance."""
    return NORMALISATION_VERSION
