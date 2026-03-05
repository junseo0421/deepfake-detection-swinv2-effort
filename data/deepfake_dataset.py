# data/deepfake_dataset.py
import random
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from utils.utils import read_rgb_frames, get_full_frame_padded


class DeepfakeDataset(Dataset):
    """
    추후에 수정
    CSV format:
      path,label
      /abs/or/rel/path/to/file.jpg,0
      /abs/or/rel/path/to/file.mp4,1

    label: Real=0, Fake=1
    """
    def __init__(
        self,
        csv_path: str,
        num_frames: int,
        target_size: Tuple[int, int],
        mode: str = "train",     # "train" or "val"
        transform=None,
        root_dir: Optional[str] = None,
        filename_col: str = "filename",
        label_col_candidates=("Ground Truth", "Video Ground Truth", "label"),
    ):
        try:
            self.df = pd.read_csv(csv_path, encoding="utf-8-sig", encoding_errors="replace")
        except UnicodeDecodeError:
            self.df = pd.read_csv(csv_path, encoding="cp949", encoding_errors="replace")

        self.num_frames = int(num_frames)
        self.target_size = target_size
        self.mode = mode
        self.transform = transform

        self.root_dir = Path(root_dir) if root_dir is not None else None

        # filename 컬럼 체크
        if filename_col not in self.df.columns:
            raise KeyError(f"CSV에 '{filename_col}' 컬럼이 없음. 실제 컬럼: {list(self.df.columns)}")
        self.filename_col = filename_col

        # label 컬럼 자동 선택
        self.label_col = None
        for c in label_col_candidates:
            if c in self.df.columns:
                self.label_col = c
                break
        if self.label_col is None:
            raise KeyError(f"CSV에 라벨 컬럼이 없음. 후보={label_col_candidates}, 실제={list(self.df.columns)}")

        fn = self.df[self.filename_col].astype(str)

        fn_clean = fn.str.split("?", n=1).str[0].str.strip()

        invalid = (
                self.df[self.filename_col].isna()
                | (fn.str.strip() == "")
                | (fn.str.upper().str.strip() == "NAN")
                | (fn.str.contains(r"#NAME\?", na=False))
                | (fn_clean.str.lower().str.endswith(".webm"))
        )

        before = len(self.df)
        self.df = self.df.loc[~invalid].reset_index(drop=True)
        after = len(self.df)

        # debug
        print(f"[DeepfakeDataset] filtered invalid filenames: {before - after} / {before}")

    def __len__(self):
        return len(self.df)

    def _pick_one_rgb(self, file_path: Path):
        frames = read_rgb_frames(file_path, num_frames=self.num_frames)  # List[np.ndarray RGB]
        if not frames:
            return None

        if self.mode == "train":
            rgb = random.choice(frames)  # 랜덤 프레임 1장
        else:
            rgb = frames[len(frames) // 2]  # 검증은 중앙 프레임
        return rgb

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = Path(str(row[self.filename_col]))

        ### debug 용도
        if self.root_dir is not None:
            p = self.root_dir / p

        # 라벨: Real=0, Fake=1
        gt = str(row[self.label_col]).strip()
        if gt.lower() in ("fake", "1", "true"):
            label = 1
        elif gt.lower() in ("real", "0", "false"):
            label = 0
        else:
            raise ValueError(f"알 수 없는 Ground Truth 값: '{gt}' (row={idx})")

        rgb = self._pick_one_rgb(p)
        if rgb is None:
            raise RuntimeError(f"Failed to read frames: {p}")

        img = get_full_frame_padded(Image.fromarray(rgb), self.target_size)  # PIL
        if self.transform is not None:
            img = self.transform(img)

        return img, label
