import argparse
import random
import zipfile
from pathlib import Path
import shutil
import re


VIDEO_EXTS_DEFAULT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}


def safe_join(root: Path, *parts: str) -> Path:
    """
    ZipSlip(../) 방지: root 밖으로 나가려는 경로면 차단
    """
    out = root.joinpath(*parts)
    resolved = out.resolve()
    root_resolved = root.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise ValueError(f"Unsafe path detected: {out}")
    return out


def infer_class_name_from_filename(member_path: str, split_char: str = "_") -> str | None:
    """
    zip 멤버 경로에서 파일명 기준으로 클래스명 추출.
    예: 'bigfolder/ABC_001.mp4' -> 'ABC'
        'bigfolder/audio_driven_01_123.mp4' -> 'audio_driven_01'  (마지막 토큰이 숫자면 앞을 전부 cls로)
    """
    fname = Path(member_path).name  # 폴더 경로 제거
    stem = Path(fname).stem         # 확장자 제거

    parts = stem.split(split_char)
    if len(parts) < 2:
        return None

    # [클래스명]_[숫자] 패턴을 우선 지원 (클래스명에 '_'가 포함될 수 있으므로 뒤에서 검사)
    if parts[-1].isdigit():
        cls = split_char.join(parts[:-1]).strip()
        return cls if cls else None

    # 뒤가 숫자가 아니면 "첫 토큰"을 클래스명으로 fallback
    cls = parts[0].strip()
    return cls if cls else None


def infer_class_name_from_path(member_path: str, class_depth: int) -> str | None:
    """
    (옵션) 기존 폴더 depth 방식도 유지하고 싶을 때 사용
    member_path 예: 'audio_driven/138487/xxx.mp4'
    class_depth=2 이면 parts[1] = '138487' 를 클래스명으로 봄
    """
    parts = member_path.split("/")
    if len(parts) <= class_depth:
        return None
    return parts[class_depth - 1]


def extract_selected_members(zf: zipfile.ZipFile, members: list[str], out_dir: Path):
    for m in members:
        if m.endswith("/"):
            continue

        filename = Path(m).name
        target_path = safe_join(out_dir, filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with zf.open(m, "r") as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def process_zip(
    zip_path: Path,
    output_root: Path,
    per_class: int,
    video_exts: set[str],
    pick_mode: str,
    rng: random.Random,
    class_source: str,
    class_depth: int,
    split_char: str,
):
    zip_stem = zip_path.stem
    out_zip_dir = output_root / zip_stem
    out_zip_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        buckets: dict[str, list[str]] = {}

        for info in zf.infolist():
            name = info.filename

            if name.endswith("/"):
                continue

            ext = Path(name).suffix.lower()
            if ext not in video_exts:
                continue

            if class_source == "filename":
                cls = infer_class_name_from_filename(name, split_char=split_char)
            else:
                cls = infer_class_name_from_path(name, class_depth=class_depth)

            if cls is None:
                continue

            buckets.setdefault(cls, []).append(name)

        extracted_total = 0
        for cls, files in buckets.items():
            if not files:
                continue

            if pick_mode == "sorted":
                chosen = sorted(files)[:per_class]
            else:
                if len(files) <= per_class:
                    chosen = files
                else:
                    chosen = rng.sample(files, per_class)

            cls_out_dir = out_zip_dir / cls
            extract_selected_members(zf, chosen, cls_out_dir)
            extracted_total += len(chosen)

    return len(buckets), extracted_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zips-root", type=str,
                        default=r'D:\contest\2026_deepfake\AIHub\딥페이크 변조 영상\딥페이크 변조 영상',
                        help="zip 파일들이 모여있는 폴더")
    parser.add_argument("--out-root", type=str,
                        default=r'D:\contest\2026_deepfake\AIHub\zip_extract_2_noise',
                        help="추출 결과 저장 폴더")

    parser.add_argument("--per-class", type=int, default=2, help="클래스당 뽑을 영상 개수 (1 또는 2 등)")
    parser.add_argument("--pick-mode", choices=["random", "sorted"], default="random",
                        help="클래스 내에서 어떤 방식으로 파일을 고를지")
    parser.add_argument("--seed", type=int, default=42, help="random 선택 재현용 seed")

    parser.add_argument("--exts", type=str, default=",".join(sorted(VIDEO_EXTS_DEFAULT)),
                        help="영상 확장자 목록(콤마구분). 예: .mp4,.avi,.mkv")
    parser.add_argument("--recursive", action="store_true", default=True, help="하위 폴더까지 zip 탐색")

    # ✅ 새로 추가: 클래스 추출 방식
    parser.add_argument("--class-source", choices=["filename", "path"], default="filename",
                        help="클래스명을 어디서 뽑을지: filename=파일명(prefix), path=기존 depth 방식")
    parser.add_argument("--split-char", type=str, default="_",
                        help="filename 방식에서 클래스명/번호를 가르는 문자 (기본: _)")
    # (path 방식 호환용) 기존 인자도 남겨둠
    parser.add_argument("--class-depth", type=int, default=2,
                        help="path 방식일 때만 사용: 클래스 폴더 depth (기본 2)")

    args = parser.parse_args()

    zips_root = Path(args.zips_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    video_exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    rng = random.Random(args.seed)

    zip_iter = zips_root.rglob("*.zip") if args.recursive else zips_root.glob("*.zip")
    zip_list = sorted(zip_iter)

    if not zip_list:
        print(f"[!] zip을 찾지 못했습니다: {zips_root}")
        return

    total_classes = 0
    total_extracted = 0

    for zp in zip_list:
        try:
            cls_cnt, ext_cnt = process_zip(
                zip_path=zp,
                output_root=out_root,
                per_class=args.per_class,
                video_exts=video_exts,
                pick_mode=args.pick_mode,
                rng=rng,
                class_source=args.class_source,
                class_depth=args.class_depth,
                split_char=args.split_char,
            )
            total_classes += cls_cnt
            total_extracted += ext_cnt
            print(f"[OK] {zp.name}: classes={cls_cnt}, extracted={ext_cnt}")
        except zipfile.BadZipFile:
            print(f"[SKIP] 손상된 zip: {zp}")
        except Exception as e:
            print(f"[ERR] {zp}: {e}")

    print(f"\n=== DONE ===")
    print(f"zip 개수: {len(zip_list)}")
    print(f"총 클래스 수(누적): {total_classes}")
    print(f"총 추출 파일 수: {total_extracted}")


if __name__ == "__main__":
    main()
