import pytest

from mtgir.cli import best_match, load_cards

from . import TEST_DATA_ROOT


@pytest.mark.parametrize(
    "test_image, expected_gid",
    [
        ("test_ulamog.jpg", "225e3cc6-34d0-4f81-9f49-162d97e2ea59"),
        ("test_darkslick.jpg", "e530388b-eb19-4211-abd8-8a4c3c38c3af"),
        ("test_darkslick_90.jpg", "e530388b-eb19-4211-abd8-8a4c3c38c3af"),
        ("test_oketra.jpg", "104503a6-bca5-48d7-88b1-424f98985d75"),
        ("test_shaku.jpg", "babe91f2-06be-4501-a95b-20968e906e1b"),
        ("test_naheb.jpg", "60c90d5d-7b3b-48d2-85f6-d6a2a452c0e9"),
    ],
)
def test_best_match(test_image: str, expected_gid: str):
    data = load_cards(update=False)
    gid = best_match(data, TEST_DATA_ROOT / test_image)
    assert gid == expected_gid
