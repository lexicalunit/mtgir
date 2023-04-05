from pathlib import Path

TST_ROOT = Path(__file__).resolve().parent
TEST_DATA_ROOT = TST_ROOT / "data"
REPO_ROOT = TST_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"

SRC_DIRS = [
    REPO_ROOT / "tests",
    SRC_ROOT / "mtgir",
]
