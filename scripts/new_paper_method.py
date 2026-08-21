"""Create a traceable implementation record for a paper PDF."""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO_ROOT / "papers" / "templates" / "method-plan.md"


def valid_slug(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
        raise argparse.ArgumentTypeError(
            "slug must contain lowercase letters/numbers separated by single hyphens"
        )
    return value


def verify_pdf(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"paper does not exist: {path}")
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError(f"paper is not a PDF: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_link(source: Path, destination_dir: Path) -> str:
    import os

    return Path(os.path.relpath(source, destination_dir)).as_posix()


def scaffold(
    paper: Path,
    slug: str,
    title: str,
    paper_id: str,
    output_root: Path,
    template_path: Path = DEFAULT_TEMPLATE,
) -> Path:
    paper = paper.resolve()
    verify_pdf(paper)
    destination = output_root / slug / "implementation.md"
    if destination.exists():
        raise FileExistsError(f"method record already exists: {destination}")

    template = template_path.read_text(encoding="utf-8")
    values = {
        "{{TITLE}}": title,
        "{{SLUG}}": slug,
        "{{PAPER_ID}}": paper_id or "not recorded",
        "{{PDF_NAME}}": paper.name,
        "{{PDF_LINK}}": relative_link(paper, destination.parent.resolve()),
        "{{SHA256}}": sha256(paper),
        "{{DATE}}": date.today().isoformat(),
    }
    for marker, value in values.items():
        template = template.replace(marker, value)

    destination.parent.mkdir(parents=True, exist_ok=False)
    destination.write_text(template, encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path, help="PDF in papers/inbox")
    parser.add_argument("--slug", required=True, type=valid_slug)
    parser.add_argument("--title", required=True)
    parser.add_argument("--paper-id", default="")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "papers" / "methods",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        destination = scaffold(
            paper=args.paper,
            slug=args.slug,
            title=args.title,
            paper_id=args.paper_id,
            output_root=args.output_root,
        )
    except (ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
    print(destination)


if __name__ == "__main__":
    main()

