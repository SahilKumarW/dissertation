"""Deterministic conversion of direction text to bearings, and distances to metres.

Both live in Python, not in the prompt. The model reports what the description
*says*; arithmetic and table lookup are done here, where they are testable and
identical on every run (master spec sections 14 and 15).

If a direction cannot be mapped, the caller abstains. Guessing a bearing turns
an interpretation failure into a confident wrong coordinate, which is the worst
possible outcome for a positional-error metric.
"""

from __future__ import annotations

import re

__all__ = [
    "BEARINGS",
    "bearing_for",
    "UNIT_METRES",
    "distance_in_metres",
    "UnknownDirection",
    "UnknownUnit",
]


class UnknownDirection(ValueError):
    """The direction text has no defensible bearing."""


class UnknownUnit(ValueError):
    """The distance unit is not one we convert."""


# 16-point compass. The 8 cardinal and intercardinal points are what the corpus
# overwhelmingly uses; the secondary intercardinals appear in older labels
# ("N.N.W. of Erua") and are included so those are not silently unresolvable.
BEARINGS: dict[str, float] = {
    "n": 0.0, "nne": 22.5, "ne": 45.0, "ene": 67.5,
    "e": 90.0, "ese": 112.5, "se": 135.0, "sse": 157.5,
    "s": 180.0, "ssw": 202.5, "sw": 225.0, "wsw": 247.5,
    "w": 270.0, "wnw": 292.5, "nw": 315.0, "nnw": 337.5,
}

_WORDS = {
    "north": "n", "south": "s", "east": "e", "west": "w",
    "nth": "n", "sth": "s",
}

_NOISE = re.compile(r"[\s\-\.,]+")


def _collapse(text: str) -> str:
    """'north-east' -> 'ne', 'N.N.W.' -> 'nnw', 'south west' -> 'sw'."""
    lowered = _NOISE.sub("", str(text).strip().lower())
    if not lowered:
        return ""
    # Spelled-out forms: replace longest words first so 'northnortheast' does
    # not become 'nnorthe' by replacing 'north' inside 'northeast' twice wrong.
    for word in sorted(_WORDS, key=len, reverse=True):
        lowered = lowered.replace(word, _WORDS[word])
    return lowered


def bearing_for(direction_text: str | None) -> float:
    """Bearing in degrees clockwise from true north.

    Accepts abbreviations and spelled forms, with or without separators:
    'SW', 's.w.', 'south-west', 'southwest', 'south west' all give 225.
    """
    if direction_text is None:
        raise UnknownDirection("no direction given")
    key = _collapse(direction_text)
    if key not in BEARINGS:
        raise UnknownDirection(f"cannot map direction to a bearing: {direction_text!r}")
    return BEARINGS[key]


# Canonical internal unit is the metre. Miles are statute miles: the corpus is
# terrestrial collection localities, so nautical miles are not in scope and are
# rejected rather than assumed.
UNIT_METRES: dict[str, float] = {
    "m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometre": 1000.0, "kilometres": 1000.0,
    "kilometer": 1000.0, "kilometers": 1000.0, "kms": 1000.0,
    "mile": 1609.344, "miles": 1609.344, "mi": 1609.344, "mls": 1609.344,
    "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
}


def distance_in_metres(value: float | None, unit: str | None) -> float:
    """Convert a distance to metres. Rejects missing, negative and unknown units."""
    if value is None:
        raise UnknownUnit("no distance value given")
    if unit is None:
        raise UnknownUnit("no distance unit given")
    key = str(unit).strip().lower().rstrip(".")
    if key not in UNIT_METRES:
        raise UnknownUnit(f"unknown distance unit: {unit!r}")
    metres = float(value) * UNIT_METRES[key]
    if metres < 0:
        raise UnknownUnit(f"negative distance: {value} {unit}")
    return metres
