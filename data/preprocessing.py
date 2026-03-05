import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from utils.utils import *


class PreprocessOutput:
    def __init__(
            self,
            filename: str,
            imgs: List[Image.Image],
            error: Optional[str] = None
    ):
        self.filename = filename
        self.imgs = imgs
        self.error = error


def preprocess_one(file_path, num_frames, target_size) -> PreprocessOutput:
    """
    파일 하나에 대한 전처리 수행

    Args:
        file_path: 처리할 파일 경로
        num_frames: 비디오에서 추출할 프레임 수

    Returns:
        PreprocessOutput 객체
    """
    try:
        frames = read_rgb_frames(file_path, num_frames=num_frames)

        imgs: List[Image.Image] = []

        for rgb in frames:
            imgs.append(get_full_frame_padded(Image.fromarray(rgb), target_size))

        return PreprocessOutput(file_path.name, imgs, None)

    except Exception as e:
        return PreprocessOutput(file_path.name, [], str(e))

