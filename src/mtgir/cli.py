from __future__ import annotations

import click
import uvicorn

from .api import app
from .ir import match_results
from .scryfall import download_cards, download_dataset, load_cards, load_db


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
    "--api",
    is_flag=True,
    default=False,
    help="run as a HTTP server",
)
@click.argument("filename", required=False)
def main(
    download: bool,
    update: bool,
    api: bool,
    filename: str | None,
) -> int:
    if api:
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return 0

    if download:
        if needs_update := download_dataset():
            update = needs_update
        download_cards()

    if filename is None:
        return 0

    data = load_cards(update)
    db = load_db()

    if not (matches := match_results(db, data, filename)):
        print("no match found")
        return 1

    for match in matches:
        print(f"[{match.score:.2f}] {match.name} -> {match.target} ({match.uri})")
    return 0
