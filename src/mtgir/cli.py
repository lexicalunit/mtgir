from __future__ import annotations

import json
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import total_ordering
from itertools import cycle
from pathlib import Path
from sys import exit
from typing import Any, Iterable, Optional, Union

import click
import cv2 as cv
import numpy as np
import requests

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# app config
CACHE_FILE = "cache"
IMAGES_DIR = "images"
LATEST_FILE = "latest"

# scryfall config
BULK_DATA_URL = "https://api.scryfall.com/bulk-data/unique-artwork"
IMAGE_PREFERENCES = ("large", "border_crop", "normal")
INVALID_SET_NAMES = {"Substitute Cards"}
INVALID_SET_TYPES = {"memorabilia"}
INVALID_TYPES = {"Basic Land"}
SCRYFALL_JSON_FILE = "unique-artwork.json"

# OpenCV config
MATCH_MODE = cv.NORM_HAMMING2
NFEATURES = 500
READ_MODE = cv.IMREAD_GRAYSCALE


def best_image_uri(image_uris: dict[str, str]) -> Optional[str]:
    for pref in IMAGE_PREFERENCES:
        if uri := image_uris.get(pref):
            return uri


def download_card(dest: Path, datum: dict[str, Any], delete: bool = False):
    if delete:
        logger.info("deleting %s...", dest)
        dest.unlink(missing_ok=True)
        return

    if dest.exists():
        logger.info("%s exists", dest)
        return

    image_uris: Optional[dict[str, str]] = datum.get("image_uris")
    if not image_uris:
        logger.warning("no image_uris in %s", datum)
        return

    uri = best_image_uri(image_uris)
    if not uri:
        logger.error("no preferential uri in %s", image_uris)
        exit(1)

    try:
        logger.info("downloading %s...", uri)
        r = requests.get(uri, allow_redirects=True)
        with open(dest, "wb") as f:
            f.write(r.content)
    except Exception as e:
        logger.exception(e)


def download_dataset() -> bool:
    """Returns True if the dataset was updated, False otherwise."""
    r = requests.get(BULK_DATA_URL)
    assert r.status_code == 200, r.content
    download_uri: Optional[str] = r.json().get("download_uri")
    assert download_uri is not None

    latest_path = Path(LATEST_FILE)
    current_latest: Optional[str] = None
    new_latest = download_uri.split("/")[-1]

    if latest_path.exists():
        assert latest_path.is_file()
        with open(latest_path, "r") as latest_file:
            current_latest = latest_file.readline().strip()

    if current_latest == new_latest:
        return False

    r = requests.get(download_uri)
    assert r.status_code == 200, r.content
    with open(SCRYFALL_JSON_FILE, "wb") as out:
        out.write(r.content)
    with open(latest_path, "w") as latest_file:
        latest_file.write(new_latest)
    return True


def download_cards():
    Path(IMAGES_DIR).mkdir(exist_ok=True)
    with open(SCRYFALL_JSON_FILE) as df:
        data = json.load(df)
        for datum in data:
            gid = datum.get("id")
            if not gid:
                logger.error("missing id in %s", datum)
                exit(1)

            delete = "paper" not in datum.get("games", []) or datum.get("digital")
            delete = delete or datum.get("set_type") in INVALID_SET_TYPES
            delete = delete or any(n in datum.get("set_name") for n in INVALID_SET_NAMES)
            if type_line := datum.get("type_line"):
                delete = delete or any(n in type_line for n in INVALID_TYPES)

            if (
                (faces := datum.get("card_faces"))
                and len(faces) == 2
                and all("image_uris" in face for face in faces)
            ):
                front = Path(IMAGES_DIR) / f"{gid}-front.jpg"
                back = Path(IMAGES_DIR) / f"{gid}-back.jpg"
                download_card(front, faces[0], delete=delete)
                download_card(back, faces[1], delete=delete)
            else:
                download_card(Path(IMAGES_DIR) / f"{gid}.jpg", datum, delete=delete)


def load_image(path: Union[str, Path]) -> np.ndarray:
    with open(path, "rb") as fin:
        np_array = np.fromfile(fin, np.uint8)
        return cv.imdecode(np_array, READ_MODE)


def load_cards(update: bool) -> dict[str, np.ndarray]:
    logger.info("loading cards...")
    data: dict[str, np.ndarray] = {}

    cache = Path(CACHE_FILE)
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
            if not update:
                return data

    orb = cv.ORB_create(nfeatures=NFEATURES)
    dir = Path(IMAGES_DIR)
    for fname in os.listdir(dir):
        gid = fname[0:-4]
        if gid in data:
            logger.info("%s is cached", gid)
            continue
        logger.info("loading %s...", gid)
        f = dir / fname
        assert f.is_file()
        img = load_image(f)
        _, descriptors = orb.detectAndCompute(img, None)
        data[gid] = descriptors

    with open(cache, "wb") as f:
        pickle.dump(data, f)

    logger.info("done loading cards")
    return data


@total_ordering
class Match:
    def __init__(self, gid: str, count: int) -> None:
        self.gid = gid
        self.count = count

    def __repr__(self) -> str:
        return f"Match<{self.gid}, {self.count}>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.gid == other
        if not isinstance(other, Match):
            return NotImplemented
        return self.gid == other.gid

    def __lt__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.gid < other
        if not isinstance(other, Match):
            return NotImplemented
        return self.gid < other.gid


def compare(gid: str, lhs: np.ndarray, rhs: np.ndarray) -> Optional[Match]:
    good_matches = 0
    for pair in cv.BFMatcher(MATCH_MODE).knnMatch(lhs, rhs, k=2):
        try:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches += 1
        except ValueError:
            pass
    if good_matches == 0:
        return None
    return Match(gid, good_matches)


def match_image(img: np.ndarray, data: dict[str, np.ndarray]) -> Iterable[Match]:
    logger.info("matching image...")
    orb = cv.ORB_create(nfeatures=NFEATURES)
    pool = ProcessPoolExecutor()
    _, tds = orb.detectAndCompute(img, None)
    gids: list[str] = []
    descriptors: list[np.ndarray] = []
    for gid, des in data.items():
        if des is not None:
            gids.append(gid)
            descriptors.append(des)
    matches = pool.map(compare, gids, cycle([tds]), descriptors)
    return (m for m in matches if m is not None)


def best_match(matches: Iterable[Match]) -> Optional[Match]:
    try:
        return max(matches, key=lambda m: m.count)
    except ValueError:
        return None


def rank_matches(matches: Iterable[Match]) -> list[Match]:
    return sorted(matches, key=lambda m: m.count, reverse=True)


def load_db() -> dict[str, Any]:
    db: dict[str, Any] = {}
    with open(SCRYFALL_JSON_FILE) as df:
        data = json.load(df)
        for datum in data:
            db[datum["id"]] = datum
    return db


def show_match(lhs: np.ndarray, rhs: np.ndarray):
    orb = cv.ORB_create(nfeatures=NFEATURES)
    lhs_keys, lhs_desc = orb.detectAndCompute(lhs, None)
    assert lhs_desc is not None
    rhs_keys, rhs_desc = orb.detectAndCompute(rhs, None)
    assert rhs_desc is not None
    matches = cv.BFMatcher(MATCH_MODE).knnMatch(lhs_desc, rhs_desc, k=2)
    good_matches = [[0, 0] for _ in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            good_matches[i] = [1, 0]
    out_img = cv.drawMatchesKnn(
        lhs,
        lhs_keys,
        rhs,
        rhs_keys,
        matches,
        outImg=None,
        matchColor=(0, 155, 0),
        singlePointColor=(0, 255, 255),
        matchesMask=good_matches,
        flags=0,
    )

    logger.info("showing matched features, press ESC to quit...")
    cv.imshow("ORB", out_img)

    while True:
        if cv.waitKey(10) & 0xFF == 27:
            break


@click.command()
@click.option(
    "--download",
    "-d",
    is_flag=True,
    default=False,
    help="download card images, image cache will update if necessary",
)
@click.option(
    "--update",
    "-u",
    is_flag=True,
    default=False,
    help="force update the image cache",
)
@click.option(
    "--show",
    "-s",
    is_flag=True,
    default=False,
    help="show the matched features between the given filename and the best match",
)
@click.argument("filename", required=False)
def main(
    download: bool,
    update: bool,
    show: bool,
    filename: Optional[str],
) -> int:
    if download:
        if needs_update := download_dataset():
            update = needs_update
        download_cards()

    data = load_cards(update)

    if filename is not None:
        img = load_image(filename)
        matches = match_image(img, data)
        best = best_match(matches)
        if best is None:
            print("I don't know what that is.")
            return 1
        if show:
            target = load_image(f"{IMAGES_DIR}/{best.gid}.jpg")
            show_match(img, target)
        else:
            db = load_db()
            name = db[best.gid]["name"]
            print(f"That's {name} ({best.gid})!")

    return 0
