import os
from pathlib import Path
import cv2
import numpy as np
from typing import Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image, ExifTags
import numpy as np

def load_image_with_exif_fix(path: Path):
    img = Image.open(path)

    try:
        for orientation in ExifTags.TAGS:
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = img._getexif()
        if exif is not None:
            ori = exif.get(orientation, None)

            if ori == 3:
                img = img.rotate(180, expand=True)
            elif ori == 6:
                img = img.rotate(270, expand=True)  # 90도 CW
            elif ori == 8:
                img = img.rotate(90, expand=True)   # 90도 CCW
    except Exception:
        pass

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


model_path = hf_hub_download(
    repo_id="AdamCodd/YOLOv11n-face-detection",
    filename="model.pt"
)

yolo_face = YOLO(model_path)


def detect_faces_yolo(img_rgb: np.ndarray, conf_thres=0.25):
    """
    img_rgb: RGB image
    returns: list of (x1, y1, x2, y2)
    """
    results = yolo_face.predict(
        img_rgb,
        imgsz=640,
        conf=conf_thres,
        verbose=False
    )

    faces = []
    for det in results:
        if det.boxes is None:
            continue
        for box in det.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            faces.append((x1, y1, x2, y2))
    return faces


def extract_face_center_crop_robust(
    img_rgb: np.ndarray,
    scales=(1.0, 1.25, 1.5),
    upsamples=(1, 2, 3),   # 의미만 유지 (YOLO는 실제 upsample 안 함)
    crop_scale: float = 1.2
) -> Optional[np.ndarray]:

    for s in scales:
        resized = cv2.resize(
            img_rgb, None,
            fx=s, fy=s,
            interpolation=cv2.INTER_LINEAR
        )

        faces = detect_faces_yolo(resized)
        if len(faces) == 0:
            continue

        # 가장 큰 얼굴 선택 (dlib 로직 그대로)
        best = None
        best_area = 0
        for (x1, y1, x2, y2) in faces:
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

        if best is None:
            continue

        x1, y1, x2, y2 = best

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
    save_path = output_dir / (img_path.stem + ".jpg")

    if save_path.exists():
        print(f"[SKIP][DONE][IMG] {img_path.name}")
        return "ok"

    img = load_image_with_exif_fix(img_path)

    if img is None:
        print(f"[SKIP] Read fail: {img_path.name}")
        return "read_fail"

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop = extract_face_center_crop_robust(img_rgb)

    if crop is None:
        print(f"[SKIP] No face: {img_path.name}")
        return "no_face"

    cv2.imwrite(str(save_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print(f"[OK][IMG] {img_path.name}")
    return "ok"


def process_video_file(video_path: Path, output_dir: Path, n_frames: int = 10):
    first_out = output_dir / f"{video_path.stem}_1.jpg"
    if first_out.exists():
        print(f"[SKIP][DONE][VID] {video_path.name}")
        return "ok"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] Video open fail: {video_path.name}")
        return "read_fail"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return "read_fail"

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    save_idx = 1
    found = False

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        crop = extract_face_center_crop_robust(frame_rgb)

        if crop is None:
            continue

        found = True
        save_path = output_dir / f"{video_path.stem}_{save_idx}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        print(f"[OK][VID] {video_path.name} → {save_path.name}")
        save_idx += 1

    cap.release()
    return "ok" if found else "no_face"


def process_folder(input_dir: str, output_dir: str, n_video_frames: int = 10):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".jfif"}
    vid_exts = {".mp4", ".mov", ".avi", ".mkv"}

    # 하위/하위의 하위 폴더까지 전부 탐색
    all_files = sorted([p for p in input_dir.rglob("*") if p.is_file()])

    stats = {
        "total": 0,
        "img_ok": 0,
        "vid_ok": 0,
        "no_face": 0,
        "read_fail": 0,
        "unsupported": 0
    }

    print(f"--- 처리 시작 | 총 {len(all_files)} 파일(재귀 포함) ---")

    for i, path in enumerate(all_files, 1):
        stats["total"] += 1

        if i % 100 == 0:
            print(f"--- 진행 {i}/{len(all_files)} ---")

        ext = path.suffix.lower()

        # 원본 폴더 구조 그대로 output에 복제
        rel = path.relative_to(input_dir)          # 예: a/b/c.jpg
        out_subdir = output_dir / rel.parent       # 예: output/a/b
        out_subdir.mkdir(parents=True, exist_ok=True)

        if ext in img_exts:
            ret = process_image_file(path, out_subdir)
            if ret == "ok":
                stats["img_ok"] += 1
            elif ret == "no_face":
                stats["no_face"] += 1
            elif ret == "read_fail":
                stats["read_fail"] += 1

        elif ext in vid_exts:
            ret = process_video_file(path, out_subdir, n_frames=n_video_frames)
            if ret == "ok":
                stats["vid_ok"] += 1
            elif ret == "no_face":
                stats["no_face"] += 1
            elif ret == "read_fail":
                stats["read_fail"] += 1

        else:
            # (jpg만 처리하고 싶으면 img_exts를 {".jpg"}로 바꾸면 됨)
            stats["unsupported"] += 1

    print("\n========== 처리 통계 ==========")
    print(f"총 파일 수        : {stats['total']}")
    print(f"이미지 처리 성공 : {stats['img_ok']}")
    print(f"비디오 처리 성공 : {stats['vid_ok']}")
    print(f"얼굴 미검출      : {stats['no_face']}")
    print(f"읽기 실패        : {stats['read_fail']}")
    print(f"미지원 확장자    : {stats['unsupported']}")
    print("================================")


if __name__ == "__main__":
    process_folder(
        input_dir=r"D:\contest\2026_deepfake\AIHub\zip_extract_2_noise",
        output_dir=r"D:\contest\2026_deepfake\ai_hub_face_crop_noise",
        n_video_frames=2
    )
