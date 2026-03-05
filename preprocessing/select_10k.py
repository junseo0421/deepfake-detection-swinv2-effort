import argparse
import math
import os
import random
import shutil
from pathlib import Path
from typing import List


IMG_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_files(src_root: Path, exts: set[str]) -> List[Path]:
    files: List[Path] = []
    # rglob은 경로가 길어질 수 있으니, 필요시 os.walk로 바꿔도 됩니다.
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def spaced_random_sample(files: List[Path], k: int, seed: int) -> List[Path]:
    """
    files를 정렬한 뒤,
    전체를 k개의 bin으로 나누고 각 bin에서 1개씩 랜덤 선택.
    """
    n = len(files)
    if k <= 0:
        raise ValueError("k는 1 이상이어야 합니다.")
    if n == 0:
        raise ValueError("선택 가능한 파일이 없습니다.")
    if k > n:
        raise ValueError(f"요청한 k={k}가 전체 파일 수 n={n}보다 큽니다.")

    rng = random.Random(seed)

    files_sorted = sorted(files, key=lambda x: str(x).lower())

    step = n / k  # 실수 step
    chosen = []
    used_idx = set()

    for i in range(k):
        a = int(math.floor(i * step))
        b = int(math.floor((i + 1) * step)) - 1
        if b < a:
            b = a

        # bin 안에서 랜덤으로 인덱스 선택 (중복 방지)
        idx = rng.randint(a, b)
        if idx in used_idx:
            # 충돌 시 근처에서 빈 인덱스 찾기
            left = idx - 1
            right = idx + 1
            found = None
            while left >= a or right <= b:
                if left >= a and left not in used_idx:
                    found = left
                    break
                if right <= b and right not in used_idx:
                    found = right
                    break
                left -= 1
                right += 1
            if found is None:
                # bin에 빈자리가 없다면 전체 범위에서 빈자리 찾기(드문 케이스)
                j = idx
                while j < n and j in used_idx:
                    j += 1
                if j >= n:
                    j = idx
                    while j >= 0 and j in used_idx:
                        j -= 1
                if j < 0 or j >= n or j in used_idx:
                    raise RuntimeError("중복 방지 실패: 유효한 인덱스를 찾지 못했습니다.")
                idx = j
            else:
                idx = found

        used_idx.add(idx)
        chosen.append(files_sorted[idx])

    return chosen


def copy_preserve_structure(src_root: Path, dst_root: Path, selected: List[Path], dry_run: bool = False) -> None:
    for src_path in selected:
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            print(f"[DRY] {src_path} -> {dst_path}")
        else:
            shutil.copy2(src_path, dst_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 데이터셋 루트 폴더")
    ap.add_argument("--dst", required=True, help="복사 대상 루트 폴더(비어있어도 됨)")
    ap.add_argument("--k", type=int, default=10_000, help="뽑을 개수 (기본 10000)")
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    ap.add_argument("--exts", default="jpg,jpeg,png,bmp,webp", help="대상 확장자 (콤마 구분)")
    ap.add_argument("--dry-run", action="store_true", help="실제 복사 없이 목록만 출력")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    exts = {"." + e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip()}

    if not src_root.exists():
        raise FileNotFoundError(f"src 폴더가 존재하지 않습니다: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 파일 수집 중... (src={src_root})")
    files = collect_files(src_root, exts if exts else IMG_EXTS_DEFAULT)
    print(f"  - 수집된 파일: {len(files):,}개")

    print(f"[2/3] 간격 랜덤 샘플링: k={args.k:,}, seed={args.seed}")
    selected = spaced_random_sample(files, args.k, args.seed)
    print(f"  - 선택된 파일: {len(selected):,}개")

    print(f"[3/3] 폴더 구조 유지 복사 시작... (dst={dst_root})")
    copy_preserve_structure(src_root, dst_root, selected, dry_run=args.dry_run)
    print("완료!")


if __name__ == "__main__":
    main()