from __future__ import annotations

from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from imagehash import ImageHash

from .ir import match_results
from .scryfall import load_cards, load_db

app = FastAPI()


@cache
def get_db() -> dict[str, Any]:
    return load_db()


@cache
def get_data() -> dict[str, ImageHash]:
    return load_cards(update=False)


@app.post("/match")
async def match(file: UploadFile = File(...)):
    now = datetime.now(timezone.utc)
    output = Path() / "matches" / f"{now.timestamp()}.jpg"

    try:
        data = get_data()
        db = get_db()

        assert data
        assert db

        if file.content_type != "image/jpeg":
            raise HTTPException(status_code=400, detail="Only JPEG files are allowed")

        contents = file.file.read()
        with Path.open(output, "wb") as f:
            f.write(contents)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Something went wrong")
    finally:
        file.file.close()

    if not (matches := match_results(db, data, output)):
        return {"message": "No matches found"}

    return {
        "message": "Matches found",
        "matches": [
            {
                "name": m.name,
                "target": m.target,
                "uri": m.uri,
                "score": m.score,
            }
            for m in matches
        ],
    }
