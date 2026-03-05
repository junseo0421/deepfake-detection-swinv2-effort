import os
from pathlib import Path
import cv2
import numpy as np
from typing import Optional
import dlib

face_detector = dlib.get_frontal_face_detector()


def extract_face_center_crop_robust(
    img_rgb: np.ndarray,
    scales=(1.0, 1.25, 1.5),
    upsamples=(1, 2, 3),
    crop_scale: float = 1.5
) -> Optional[np.ndarray]:

    for s in scales:
        resized = cv2.resize(
            img_rgb, None,
            fx=s, fy=s,
            interpolation=cv2.INTER_LINEAR
        )

        for u in upsamples:
            faces = face_detector(resized, u)
            if len(faces) == 0:
                continue

            face = max(faces, key=lambda r: r.width() * r.height())

            x1, y1, x2, y2 = (
                face.left(), face.top(),
                face.right(), face.bottom()
            )

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            bw = x2 - x1
            bh = y2 - y1
            face_size = max(bw, bh)

            crop_size = int(face_size * crop_scale)
            half = crop_size // 2

            crop_x1 = cx - half
            crop_y1 = cy - half
            crop_x2 = cx + half
            crop_y2 = cy + half

            pad_left = max(0, -crop_x1)
            pad_top = max(0, -crop_y1)
            pad_right = max(0, crop_x2 - resized.shape[1])
            pad_bottom = max(0, crop_y2 - resized.shape[0])

            if pad_left or pad_top or pad_right or pad_bottom:
                resized = cv2.copyMakeBorder(
                    resized,
                    pad_top, pad_bottom,
                    pad_left, pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )
                crop_x1 += pad_left
                crop_x2 += pad_left
                crop_y1 += pad_top
                crop_y2 += pad_top

            crop = resized[crop_y1:crop_y2, crop_x1:crop_x2]

            if s != 1.0:
                inv = 1.0 / s
                crop = cv2.resize(
                    crop,
                    (int(crop.shape[1] * inv),
                     int(crop.shape[0] * inv)),
                    interpolation=cv2.INTER_LINEAR
                )

            return crop

    return None


def process_image_file(img_path: Path, output_dir: Path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[SKIP] Read fail: {img_path.name}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    crop = extract_face_center_crop_robust(img_rgb)

    if crop is None:
        print(f"[SKIP] No face: {img_path.name}")
        return

    # 🔥 무조건 jpg로 저장
    save_name = img_path.stem + ".jpg"
    save_path = output_dir / save_name

    ok = cv2.imwrite(
        str(save_path),
        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    )

    if ok:
        print(f"[OK][IMG] {img_path.name} → {save_name}")
    else:
        print(f"[FAIL] Save error: {save_path}")


def process_video_file(
    video_path: Path,
    output_dir: Path,
    n_frames: int = 10
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] Video open fail: {video_path.name}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"[SKIP] Invalid frame count: {video_path.name}")
        cap.release()
        return

    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    save_idx = 1

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        crop = extract_face_center_crop_robust(frame_rgb)

        if crop is None:
            continue

        save_name = f"{video_path.stem}_{save_idx}.jpg"
        save_path = output_dir / save_name

        save_path.parent.mkdir(parents=True, exist_ok=True)

        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        ok = cv2.imwrite(str(save_path), bgr)

        if ok:
            print(f"[OK][VID] {video_path.name} → {save_name}")
            save_idx += 1
        else:
            print(f"[FAIL] Save error: {video_path.name} → {save_name}")

    cap.release()

def process_folder(input_dir: str, output_dir: str, n_video_frames: int = 10):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".jfif"}
    vid_exts = {".mp4", ".mov", ".avi", ".mkv"}

    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()

        # 원본 폴더 기준 상대 경로
        rel = path.relative_to(input_dir)  # 예: class1/subA/xxx.mp4
        output_subdir = output_dir / rel.parent  # 예: output/class1/subA
        output_subdir.mkdir(parents=True, exist_ok=True)

        if ext in img_exts:
            process_image_file(path, output_subdir)

        elif ext in vid_exts:
            process_video_file(path, output_subdir, n_frames=n_video_frames)

        else:
            # 폴더는 위에서 걸러졌고, 파일 중 지원 안 되는 확장자만 스킵
            print(f"[SKIP] Unsupported: {rel}")


if __name__ == "__main__":
    process_folder(
        input_dir=r"D:\contest\2026_deepfake\AIHub\zip_extract_2_noise",
        output_dir=r"D:\contest\2026_deepfake\ai_hub_face_crop_noise",
        n_video_frames=2
    )
