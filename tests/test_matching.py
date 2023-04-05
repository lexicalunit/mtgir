import pytest
from mtgir.cli import Match, load_cards, load_image, match_image, rank_matches

from . import TEST_DATA_ROOT


@pytest.mark.parametrize(
    "test_image, expected_gid",
    [
        pytest.param("test1.jpg", "225e3cc6-34d0-4f81-9f49-162d97e2ea59", id="ulamog"),
        pytest.param("test1.2.jpg", "225e3cc6-34d0-4f81-9f49-162d97e2ea59", id="ulamog.2"),
        pytest.param("test2.jpg", "e530388b-eb19-4211-abd8-8a4c3c38c3af", id="darkslick"),
        pytest.param("test2.2.jpg", "e530388b-eb19-4211-abd8-8a4c3c38c3af", id="darkslick.2"),
        pytest.param("test3.jpg", "104503a6-bca5-48d7-88b1-424f98985d75", id="oketra"),
        pytest.param("test3.2.jpg", "104503a6-bca5-48d7-88b1-424f98985d75", id="oketra.2"),
        pytest.param("test4.jpg", "babe91f2-06be-4501-a95b-20968e906e1b", id="shaku"),
        pytest.param("test5.jpg", "60c90d5d-7b3b-48d2-85f6-d6a2a452c0e9", id="neheb"),
    ],
)
def test_match_image(test_image: str, expected_gid: str):
    data = load_cards(update=False)
    img = load_image(TEST_DATA_ROOT / test_image)
    matches = match_image(img, data)
    ranks = rank_matches(matches)
    assert ranks.index(Match(expected_gid, 0)) == 0
