"""Prepare the three scCANDI datasets from the verified CFGen archive.

Download the archive explicitly with --download. Otherwise use an existing ZIP
or extracted datasets directory. Only the six requested AnnData files are extracted.
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data.single_cell import ARCHIVE_MD5, ARCHIVE_URL, DATASETS
from core.data.single_cell.prepare import prepare


def extract(archive, destination, datasets):
    wanted = {
        f"datasets/{name}/{name}_{split}.h5ad"
        for name in datasets
        for split in ("train", "test")
    }
    with zipfile.ZipFile(archive) as source:
        missing = wanted - set(source.namelist())
        if missing:
            raise ValueError(f"Archive is missing {sorted(missing)}")
        for member in sorted(wanted):
            target = destination / member
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            with source.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return destination / "datasets"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-dir",
        type=Path,
        help="Directory containing pbmc3k/, dentategyrus/, pbmc_covid/",
    )
    source.add_argument("--archive", type=Path)
    source.add_argument(
        "--download", action="store_true", help="Download the 15.1 GB CFGen archive"
    )
    parser.add_argument(
        "--download-dir", type=Path, default=Path("core/data_cache/single_cell_source")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("core/data_cache/single_cell")
    )
    parser.add_argument(
        "--layer",
        default="X_counts",
        help="Use X explicitly only if it contains raw integer counts",
    )
    parser.add_argument("--count-cap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    archive = args.archive
    if args.download:
        args.download_dir.mkdir(parents=True, exist_ok=True)
        archive = args.download_dir / "datasets.zip"
        if not archive.exists():
            temporary = archive.with_suffix(".zip.partial")
            urllib.request.urlretrieve(ARCHIVE_URL, temporary)
            temporary.replace(archive)
    if archive:
        digest = hashlib.md5()
        with archive.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != ARCHIVE_MD5:
            parser.error("CFGen archive checksum mismatch")
        directory = extract(archive, args.download_dir / "extracted", args.datasets)
    else:
        directory = args.source_dir
    for name in args.datasets:
        result = prepare(
            name,
            directory / name / f"{name}_train.h5ad",
            directory / name / f"{name}_test.h5ad",
            args.output / name,
            layer=args.layer,
            count_cap=args.count_cap,
            seed=args.seed,
        )
        print(
            f"{name}: {result['num_genes']} genes, vocabulary={result['vocab_size']}, {result['splits']}"
        )


if __name__ == "__main__":
    main()
