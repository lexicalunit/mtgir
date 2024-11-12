from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from sys import exit
from typing import Any

import cv2 as cv
import numpy as np
import requests
from imagehash import ImageHash, phash
from PIL import Image

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# app config
CACHE_FILE = "cache"
IMAGES_DIR = "images"
LATEST_FILE = "latest"

# scryfall config
# BULK_DATA_URL = "https://api.scryfall.com/bulk-data/unique-artwork"  # half the size
BULK_DATA_URL = "https://api.scryfall.com/bulk-data/default-cards"
IMAGE_PREFERENCES = ("large", "border_crop", "normal")
INVALID_SET_NAMES = {"Substitute Cards"}
INVALID_SET_TYPES = {"memorabilia"}
INVALID_TYPES = {"Basic Land", "Vanguard"}
SCRYFALL_JSON_FILE = "db.json"

CLAHE: cv.CLAHE = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def best_image_uri(image_uris: dict[str, str]) -> str | None:
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

    image_uris: dict[str, str] | None = datum.get("image_uris")
    if not image_uris:
        logger.warning("no image_uris in %s", datum)
        return

    uri = best_image_uri(image_uris)
    if not uri:
        logger.error("no preferential uri in %s", image_uris)
        exit(1)

    try:
        logger.info("downloading %s ...", uri)
        r = requests.get(uri, allow_redirects=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(r.content)
    except Exception as e:
        logger.exception(e)


def download_dataset() -> bool:
    """Returns True if the dataset was updated, False otherwise."""
    r = requests.get(BULK_DATA_URL)
    assert r.status_code == 200, r.content
    download_uri: str | None = r.json().get("download_uri")
    assert download_uri is not None

    latest_path = Path(LATEST_FILE)
    current_latest: str | None = None
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


def base_image_path(gid: str) -> Path:
    return Path(IMAGES_DIR) / gid[0] / gid[1]


def download_cards():
    Path(IMAGES_DIR).mkdir(exist_ok=True)
    with open(SCRYFALL_JSON_FILE) as df:
        arts = json.load(df)
        for datum in arts:
            gid = datum.get("id")
            if not gid:
                logger.error("missing id in %s", datum)
                exit(1)

            games = set(datum.get("games", []))
            # TODO: Maybe we can delete digital now that we're using the default db?
            delete = not games.intersection({"paper", "mtgo"})  # or datum.get("digital")
            delete = delete or datum.get("set_type") in INVALID_SET_TYPES
            delete = delete or any(n in datum.get("set_name") for n in INVALID_SET_NAMES)
            if type_line := datum.get("type_line"):
                delete = delete or any(n in type_line for n in INVALID_TYPES)

            if (
                (faces := datum.get("card_faces"))
                and len(faces) == 2
                and all("image_uris" in face for face in faces)
            ):
                front = base_image_path(gid) / f"{gid}-front.jpg"
                back = base_image_path(gid) / f"{gid}-back.jpg"
                download_card(front, faces[0], delete=delete)
                download_card(back, faces[1], delete=delete)
            else:
                download_card(base_image_path(gid) / f"{gid}.jpg", datum, delete=delete)


def hash_card(path: Path | str) -> ImageHash:
    img = cv.imread(str(path))
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    lightness, redness, yellowness = cv.split(lab)
    corrected_lightness = CLAHE.apply(lightness)
    limg = cv.merge((corrected_lightness, redness, yellowness))
    adjust = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    arr = np.uint8(255 * cv.cvtColor(adjust, cv.COLOR_BGR2RGB))
    return phash(Image.fromarray(arr), hash_size=32)


def load_cards(update: bool) -> dict[str, ImageHash]:
    logger.info("loading cards...")
    data: dict[str, ImageHash] = {}

    cache = Path(CACHE_FILE)
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
            if not update:
                return data

    dir = Path(IMAGES_DIR)

    for root, _, files in os.walk(dir):
        for fname in files:
            if not fname.endswith(".jpg"):
                continue
            gid = fname[0:-4]
            if gid in data:
                logger.info("%s is cached", gid)
                continue
            logger.info("loading %s ...", gid)
            f = Path(root) / fname
            assert f.is_file()
            data[gid] = hash_card(f)

    with open(cache, "wb") as f:
        pickle.dump(data, f)

    logger.info("done loading cards")
    return data


def load_db() -> dict[str, Any]:
    db: dict[str, Any] = {}
    with open(SCRYFALL_JSON_FILE) as df:
        data = json.load(df)
        for datum in data:
            db[datum["id"]] = datum
    return db
