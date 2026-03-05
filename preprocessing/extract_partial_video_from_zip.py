import argparse
import random
import zipfile
from pathlib import Path
import shutil


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


def infer_class_name(member_path: str, class_depth: int) -> str | None:
    """
    member_path 예: 'audio_driven/138487/xxx.mp4'
    class_depth=2 이면 parts[1] = '138487' 를 클래스명으로 봄
    """
    parts = member_path.split("/")
    if len(parts) <= class_depth:
        return None
    return parts[class_depth - 1]


def extract_selected_members(zf: zipfile.ZipFile, members: list[str], out_dir: Path):
    for m in members:
        # 디렉토리 엔트리 스킵
        if m.endswith("/"):
            continue

        # zip 내부 경로에서 파일명만 저장 (클래스 폴더 밑에 그대로 두고 싶으면 더 확장 가능)
        filename = Path(m).name
        target_path = safe_join(out_dir, filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 스트리밍 추출 (전체 zip 해제 X)
        with zf.open(m, "r") as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def process_zip(
    zip_path: Path,
    output_root: Path,
    per_class: int,
    class_depth: int,
    video_exts: set[str],
    pick_mode: str,
    rng: random.Random,
):
    zip_stem = zip_path.stem
    out_zip_dir = output_root / zip_stem
    out_zip_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # 클래스별로 영상 파일 모으기
        buckets: dict[str, list[str]] = {}

        for info in zf.infolist():
            name = info.filename

            # 폴더 스킵
            if name.endswith("/"):
                continue

            # 확장자 필터
            ext = Path(name).suffix.lower()
            if ext not in video_exts:
                continue

            cls = infer_class_name(name, class_depth=class_depth)
            if cls is None:
                continue

            buckets.setdefault(cls, []).append(name)

        # 클래스별로 N개 선택 후 추출
        extracted_total = 0
        for cls, files in buckets.items():
            if not files:
                continue

            if pick_mode == "sorted":
                chosen = sorted(files)[:per_class]
            else:
                # random
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
    parser.add_argument("--class-depth", type=int, default=1,
                        help="클래스 폴더가 zip 내부 경로의 몇 번째 depth인지 (기본 2: 예 audio_driven/클래스/파일)")
    parser.add_argument("--pick-mode", choices=["random", "sorted"], default="random",
                        help="클래스 내에서 어떤 방식으로 파일을 고를지")
    parser.add_argument("--seed", type=int, default=42, help="random 선택 재현용 seed")
    parser.add_argument("--exts", type=str, default=",".join(sorted(VIDEO_EXTS_DEFAULT)),
                        help="영상 확장자 목록(콤마구분). 예: .mp4,.avi,.mkv")
    parser.add_argument("--recursive", action="store_true", default=True, help="하위 폴더까지 zip 탐색")
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
                class_depth=args.class_depth,
                video_exts=video_exts,
                pick_mode=args.pick_mode,
                rng=rng,
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
