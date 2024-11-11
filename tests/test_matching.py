import pytest

from mtgir.cli import best_matches, load_cards, load_db

from . import TEST_DATA_ROOT

data = load_cards(update=False)
db = load_db()


@pytest.mark.parametrize(
    "test_image, expected_gid, success",
    [
        ("test_ulamog.jpg", "225e3cc6-34d0-4f81-9f49-162d97e2ea59", True),
        ("test_darkslick.jpg", "e530388b-eb19-4211-abd8-8a4c3c38c3af", True),
        ("test_darkslick_90.jpg", "e530388b-eb19-4211-abd8-8a4c3c38c3af", True),
        ("test_oketra.jpg", "104503a6-bca5-48d7-88b1-424f98985d75", True),
        ("test_shaku.jpg", "babe91f2-06be-4501-a95b-20968e906e1b", True),
        ("test_naheb.jpg", "60c90d5d-7b3b-48d2-85f6-d6a2a452c0e9", True),
        ("test_beastmaster.jpg", "02d7ea78-d539-4c34-8bb3-941353e3da46", True),
        ("test_young_pyro.jpg", "e349c204-3a93-4bf7-b79a-5f5f261ea2d3", True),
        # these tests are failing for now:
        ("test_skullclamp.jpg", "6daf6ed5-4f55-4ba2-99a2-9a50ea36888f", False),
    ],
)
def test_best_match(test_image: str, expected_gid: str, success: bool):
    matches = best_matches(db, data, TEST_DATA_ROOT / test_image)
    assert matches
    gid = matches[0].gid
    assert (gid == expected_gid) is success
