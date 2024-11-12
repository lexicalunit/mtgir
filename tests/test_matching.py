from __future__ import annotations

from typing import Any

import pytest
from imagehash import ImageHash

from mtgir.ir import best_matches
from mtgir.scryfall import load_cards, load_db

from . import TEST_DATA_ROOT


@pytest.fixture(scope="session")
def db() -> dict[str, Any]:
    result = load_db()
    assert result
    return result


@pytest.fixture(scope="session")
def data() -> dict[str, ImageHash]:
    result = load_cards(update=False)
    assert result
    return result


@pytest.mark.parametrize(
    "test_image, expected_name, success",
    [
        ("test_ulamog.jpg", "Ulamog, the Infinite Gyre", True),
        ("test_darkslick.jpg", "Darkslick Shores", True),
        ("test_darkslick_90.jpg", "Darkslick Shores", True),
        ("test_oketra.jpg", "Oketra's Monument", True),
        ("test_shaku.jpg", "Honor-Worn Shaku", True),
        ("test_neheb.jpg", "Neheb, Dreadhorde Champion", True),
        ("test_beastmaster.jpg", "Beastmaster Ascension", True),
        ("test_young_pyro.jpg", "Young Pyromancer", True),
        ("test_valakut_exploration.jpg", "Valakut Exploration", True),
        # For some reason I just can't get skullclamp to work at all,
        # even with pristine input images. I'm not sure what's going on.
        ("test_skullclamp.jpg", "Skullclamp", False),
    ],
)
def test_best_match(
    test_image: str,
    expected_name: str,
    success: bool,
    db: dict[str, Any],
    data: dict[str, ImageHash],
):
    matches = best_matches(db, data, TEST_DATA_ROOT / test_image)
    assert matches
    name = matches[0].name
    assert (name == expected_name) is success
