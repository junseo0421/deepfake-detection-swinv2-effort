from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

REAL_KEYWORDS = ["real"]   # 경로 어딘가에 포함되면 real(0)
FAKE_VALUE = 1
REAL_VALUE = 0

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DEFAULT_EXTS = IMAGE_EXTS | VIDEO_EXTS


def path_contains_any_keyword(p: Path, keywords: Iterable[str], case_insensitive: bool = True) -> bool:
    parts = list(p.parts)
    if case_insensitive:
        parts = [s.casefold() for s in parts]
        keywords = [k.casefold() for k in keywords]
    return any(any(k in part for k in keywords) for part in parts)


def make_label_from_relpath(rel_path: str) -> int:
    s = rel_path.casefold()
    return 0 if any(k in s for k in REAL_KEYWORDS) else 1


def iter_media_files(root: Path, exts: Optional[set[str]] = None) -> Iterable[Path]:
    exts = exts or DEFAULT_EXTS
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p



def build_metadata_for_one_dataset(dataset_root: Path, base_path: Path):
    dataset_name = dataset_root.name
    rows = []

    prefix_base = base_path.parent  # deepfake_processed의 부모

    for f in iter_media_files(dataset_root):
        rel_with_top = f.relative_to(prefix_base).as_posix()  # deepfake_processed/... 포함
        label = make_label_from_relpath(rel_with_top)         # <- 여기서 라벨 판별!
        rows.append((rel_with_top, label, dataset_name))

    return rows


def build_metadata_csv(
    base_dir: str,
    out_csv: str,
    only_datasets: Optional[List[str]] = None,
    exts: Optional[set[str]] = None,
) -> int:
    base_path = Path(base_dir).resolve()
    out_path = Path(out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_dirs = [p for p in base_path.iterdir() if p.is_dir()]
    if only_datasets is not None:
        only_set = set(only_datasets)
        dataset_dirs = [p for p in dataset_dirs if p.name in only_set]

    all_rows: List[Tuple[str, int, str]] = []
    for ds_root in sorted(dataset_dirs):
        all_rows.extend(build_metadata_for_one_dataset(ds_root, base_path))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "dataset"])
        w.writerows(all_rows)

    return len(all_rows)


if __name__ == "__main__":
    base_dir = r"D:\contest\2026_deepfake\ffdq_sampled_10k"     # 최상위 폴더(= deepfake_processed)
    out_csv = r"D:\contest\2026_deepfake\ffdq_sampled_10k.csv"

    n = build_metadata_csv(base_dir, out_csv)
    print(f"Saved {n} rows -> {out_csv}")
