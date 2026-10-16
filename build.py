"""Download the NLTK resources required by the deployed prediction API."""

from __future__ import annotations

from pathlib import Path

import nltk


TARGET_DIRECTORY = Path(__file__).resolve().parent / "nltk_data"


def main() -> None:
    """Place required NLTK data in the project bundle."""
    TARGET_DIRECTORY.mkdir(exist_ok=True)

    for package_name in ("stopwords", "wordnet", "omw-1.4"):
        success = nltk.download(package_name, download_dir=str(TARGET_DIRECTORY), quiet=True)
        if not success:
            raise RuntimeError(f"Unable to download NLTK resource: {package_name}")


if __name__ == "__main__":
    main()
