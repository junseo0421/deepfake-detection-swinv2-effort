# sample_real_fake_ffhq.py
# ------------------------------------------------------------
# 목표 구성
#   fake 10,000장  : fake_dir 내 "src_tgt.jpg" 파일들에서 샘플링(소스별 균형 옵션)
#   real 10,000장  :
#     - 연결 real 6,000장  : 샘플된 fake의 src 또는 tgt ID에 대응되는 FFHQ real(.png)에서 선택
#     - 비연결 real 4,000장: 연결에 쓰지 않은 FFHQ real에서 랜덤 선택
#
# 가정
#   - fake 파일명: "{src}_{tgt}.jpg" (정수 src, tgt)
#   - real 파일명: "00000.png" 같은 zero-pad 형식 (기본 폭 5)
#
# 출력
#   - out_dir/fake_selected.csv  (path, label=1)
#   - out_dir/real_selected.csv  (path, label=0, subset=connected|unconnected)
#   - out_dir/selected_all.csv   (path, label, subset)
#
# 옵션
#   --balanced_by_source : fake를 src(왼쪽 숫자) 기준으로 최대한 균형있게 뽑음
#   --copy_files         : 선택된 파일을 out_dir/fake, out_dir/real 로 복사
# ------------------------------------------------------------

import argparse
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

FAKE_RE = re.compile(r"^(\d+)_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def parse_fake_name(p: Path) -> Optional[Tuple[int, int]]:
    m = FAKE_RE.match(p.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def id_to_real_filename(img_id: int, pad: int = 5, ext: str = ".png") -> str:
    return f"{img_id:0{pad}d}{ext}"

def list_images(root: Path, exts: Set[str]) -> List[Path]:
    exts_l = {e.lower() for e in exts}
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts_l]

def balanced_sample_by_source(
    fake_paths: List[Path],
    want_n: int,
    seed: int,
) -> List[Path]:
    """
    src(왼쪽 ID) 기준으로 '가능한 한' 균형 있게 샘플링.
    - 각 src 그룹에서 라운드로빈처럼 하나씩 뽑아 want_n 채움.
    - 그룹 내 순서는 셔플.
    """
    set_seed(seed)

    buckets: Dict[int, List[Path]] = {}
    for p in fake_paths:
        parsed = parse_fake_name(p)
        if parsed is None:
            continue
        src, _ = parsed
        buckets.setdefault(src, []).append(p)

    # 그룹 내 셔플
    for src in buckets:
        random.shuffle(buckets[src])

    sources = list(buckets.keys())
    random.shuffle(sources)

    picked: List[Path] = []
    # 라운드로빈
    while len(picked) < want_n and sources:
        next_sources = []
        for src in sources:
            if len(picked) >= want_n:
                break
            if buckets[src]:
                picked.append(buckets[src].pop())
            if buckets[src]:
                next_sources.append(src)
        sources = next_sources

    return picked

def random_sample(fake_paths: List[Path], want_n: int, seed: int) -> List[Path]:
    set_seed(seed)
    if want_n >= len(fake_paths):
        return list(fake_paths)
    return random.sample(fake_paths, want_n)

def choose_connected_real_ids(
    sampled_fake: List[Path],
    want_connected_n: int,
    choose_mode: str,   # "alternate" | "random"
    seed: int,
) -> List[int]:
    """
    fake마다 src 또는 tgt 중 하나만 선택해서 real id 후보를 만듦.
    - alternate: 번갈아(src,tgt,src,tgt,...)
    - random   : fake마다 랜덤(src 또는 tgt)
    중복 제거(set) 후, 부족하면 src/tgt 풀을 이용해 추가로 채움.
    """
    set_seed(seed)

    pairs: List[Tuple[int, int]] = []
    for p in sampled_fake:
        parsed = parse_fake_name(p)
        if parsed is None:
            continue
        pairs.append(parsed)

    chosen: List[int] = []
    for i, (src, tgt) in enumerate(pairs):
        if choose_mode == "alternate":
            chosen.append(src if (i % 2 == 0) else tgt)
        else:
            chosen.append(src if random.random() < 0.5 else tgt)

    chosen_set: Set[int] = set(chosen)

    # 1차 후보가 부족하면, pairs의 src/tgt 전부에서 보충
    if len(chosen_set) < want_connected_n:
        pool = []
        for src, tgt in pairs:
            pool.append(src)
            pool.append(tgt)
        random.shuffle(pool)
        for x in pool:
            chosen_set.add(x)
            if len(chosen_set) >= want_connected_n:
                break

    # 그래도 부족하면, 남은 건 나중에 unconnected에서 채우지 않고
    # connected를 가능한 만큼만 구성하도록 반환
    return list(chosen_set)

def filter_existing_real_ids(real_dir: Path, real_ids: List[int], pad: int, ext: str) -> List[int]:
    exist = []
    for rid in real_ids:
        p = real_dir / id_to_real_filename(rid, pad=pad, ext=ext)
        if p.exists():
            exist.append(rid)
    return exist

def sample_unconnected_real(
    real_dir: Path,
    exclude_ids: Set[int],
    want_n: int,
    pad: int,
    ext: str,
    seed: int,
) -> List[int]:
    set_seed(seed)
    # real_dir의 파일명을 기반으로 가능한 id 목록 구성
    # (zero-pad 폭이 달라도 int로 파싱 가능)
    ids: List[int] = []
    for p in real_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ext.lower():
            continue
        stem = p.stem
        if stem.isdigit():
            rid = int(stem)
            if rid not in exclude_ids:
                ids.append(rid)

    if len(ids) <= want_n:
        return ids

    return random.sample(ids, want_n)

def write_csv(rows: List[Tuple[str, int, str]], out_csv: Path):
    # columns: path,label,subset
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("path,label,subset\n")
        for path, label, subset in rows:
            f.write(f"{path},{label},{subset}\n")

def maybe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_dir", type=str, required=True, help="fake 이미지 폴더 (예: flickr_deepfake)")
    ap.add_argument("--real_dir", type=str, required=True, help="FFHQ real 이미지 폴더 (예: images1024x1024)")
    ap.add_argument("--out_dir", type=str, required=True, help="출력 폴더")
    ap.add_argument("--fake_n", type=int, default=10000)
    ap.add_argument("--real_connected_n", type=int, default=6000)
    ap.add_argument("--real_unconnected_n", type=int, default=4000)

    ap.add_argument("--balanced_by_source", action="store_true", help="fake를 src 기준으로 균형 샘플링")
    ap.add_argument("--choose_mode", type=str, default="alternate", choices=["alternate", "random"],
                    help="connected real을 뽑을 때 fake마다 src/tgt 선택 방식")
    ap.add_argument("--real_pad", type=int, default=5, help="real 파일명 zero-pad 폭 (기본 5: 00000.png)")
    ap.add_argument("--real_ext", type=str, default=".png", help="real 확장자 (기본 .png)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--copy_files", action="store_true", help="선택된 파일들을 out_dir/fake, out_dir/real로 복사")
    args = ap.parse_args()

    set_seed(args.seed)

    fake_dir = Path(args.fake_dir)
    real_dir = Path(args.real_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fake_all = []
    for p in fake_dir.iterdir():
        if not p.is_file():
            continue
        if parse_fake_name(p) is None:
            continue
        fake_all.append(p)

    if len(fake_all) == 0:
        raise RuntimeError("fake_dir에서 '{src}_{tgt}.jpg' 형식 파일을 찾지 못했습니다.")

    if args.balanced_by_source:
        fake_sel = balanced_sample_by_source(fake_all, args.fake_n, seed=args.seed)
    else:
        fake_sel = random_sample(fake_all, args.fake_n, seed=args.seed)

    connected_ids_raw = choose_connected_real_ids(
        sampled_fake=fake_sel,
        want_connected_n=args.real_connected_n,
        choose_mode=args.choose_mode,
        seed=args.seed,
    )

    # 실제로 real_dir에 존재하는 것만 필터링
    connected_ids_exist = filter_existing_real_ids(
        real_dir=real_dir,
        real_ids=connected_ids_raw,
        pad=args.real_pad,
        ext=args.real_ext,
    )

    # 존재하는 id가 목표보다 많으면 랜덤으로 6,000으로 맞춤
    if len(connected_ids_exist) > args.real_connected_n:
        connected_ids = random.sample(connected_ids_exist, args.real_connected_n)
    else:
        connected_ids = connected_ids_exist

    connected_set = set(connected_ids)

    unconnected_ids = sample_unconnected_real(
        real_dir=real_dir,
        exclude_ids=connected_set,
        want_n=args.real_unconnected_n,
        pad=args.real_pad,
        ext=args.real_ext,
        seed=args.seed + 1,
    )

    # 만약 connected가 부족해서 real 총합이 1만이 안 되면,
    # unconnected에서 추가로 더 채우기
    total_real_want = args.real_connected_n + args.real_unconnected_n
    real_now = len(connected_ids) + len(unconnected_ids)
    if real_now < total_real_want:
        need = total_real_want - real_now
        extra = sample_unconnected_real(
            real_dir=real_dir,
            exclude_ids=connected_set.union(set(unconnected_ids)),
            want_n=need,
            pad=args.real_pad,
            ext=args.real_ext,
            seed=args.seed + 2,
        )
        unconnected_ids = unconnected_ids + extra

    fake_rows: List[Tuple[str, int, str]] = []
    real_rows: List[Tuple[str, int, str]] = []
    all_rows: List[Tuple[str, int, str]] = []

    # fake rows
    for p in fake_sel:
        fake_rows.append((str(p.resolve()), 1, "fake"))
        all_rows.append((str(p.resolve()), 1, "fake"))

    # real rows (connected)
    for rid in connected_ids:
        rp = real_dir / id_to_real_filename(rid, pad=args.real_pad, ext=args.real_ext)
        real_rows.append((str(rp.resolve()), 0, "connected"))
        all_rows.append((str(rp.resolve()), 0, "connected"))

    # real rows (unconnected)
    for rid in unconnected_ids:
        rp = real_dir / id_to_real_filename(rid, pad=args.real_pad, ext=args.real_ext)
        if not rp.exists():
            continue
        real_rows.append((str(rp.resolve()), 0, "unconnected"))
        all_rows.append((str(rp.resolve()), 0, "unconnected"))

    write_csv(fake_rows, out_dir / "fake_selected.csv")
    write_csv(real_rows, out_dir / "real_selected.csv")
    write_csv(all_rows, out_dir / "selected_all.csv")

    # 6) 선택 파일 복사(옵션)
    if args.copy_files:
        fake_out = out_dir / "fake"
        real_out = out_dir / "real"

        for p in fake_sel:
            maybe_copy(p, fake_out / p.name)

        for rid in connected_ids:
            rp = real_dir / id_to_real_filename(rid, pad=args.real_pad, ext=args.real_ext)
            if rp.exists():
                maybe_copy(rp, real_out / ("connected_" + rp.name))

        for rid in unconnected_ids:
            rp = real_dir / id_to_real_filename(rid, pad=args.real_pad, ext=args.real_ext)
            if rp.exists():
                maybe_copy(rp, real_out / ("unconnected_" + rp.name))

    print("============================================")
    print(f"Fake selected: {len(fake_sel)}")
    print(f"Real connected selected: {len(connected_ids)}")
    print(f"Real unconnected selected: {len(unconnected_ids)}")
    print(f"CSV saved to: {out_dir}")
    if args.copy_files:
        print(f"Files copied to: {out_dir/'fake'} and {out_dir/'real'}")
    print("============================================")


if __name__ == "__main__":
    main()