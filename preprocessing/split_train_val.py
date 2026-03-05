import os
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

CSV_ROOT = r"D:\contest\2026_deepfake\metadata_csvs"   # 여러 csv들이 들어있는 폴더(하위폴더 포함 가능)
OUT_DIR = r"D:\contest\2026_deepfake"
SEED = 42
TRAIN_RATIO = 0.8

# 어떤 컬럼을 "데이터셋 식별자"로 쓸지:
# 1) CSV 내부에 dataset 식별 컬럼이 있으면 여기에 이름 적기 (예: "dataset_name")
# 2) 없으면 None으로 두고, csv 파일명(또는 상위폴더명)에서 자동 생성
DATASET_ID_COL = None  # 예: "dataset_name"

# split 컬럼 이름(새로 만들거나 덮어씀)
SPLIT_COL = "split"

# 기대 컬럼들
FILENAME_COL = "filename"
LABEL_COL = "label"


def infer_dataset_id(csv_path: Path) -> str:
    # 상위 폴더명을 쓰고 싶으면: csv_path.parent.name
    # 파일명을 쓰고 싶으면: csv_path.stem
    return csv_path.stem


csv_paths = sorted(Path(CSV_ROOT).rglob("*.csv"))
if not csv_paths:
    raise FileNotFoundError(f"CSV를 찾을 수 없음: {CSV_ROOT}")

dfs = []
for p in csv_paths:
    df = pd.read_csv(p)

    # 필수 컬럼 체크
    for col in [FILENAME_COL, LABEL_COL]:
        if col not in df.columns:
            raise ValueError(f"[{p}] '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    # dataset 식별자 컬럼 만들기/확인
    if DATASET_ID_COL is None:
        df["dataset_id"] = infer_dataset_id(p)
        dataset_id_col = "dataset_id"
    else:
        if DATASET_ID_COL not in df.columns:
            raise ValueError(f"[{p}] DATASET_ID_COL='{DATASET_ID_COL}' 컬럼이 없습니다.")
        dataset_id_col = DATASET_ID_COL

    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)


def split_one_dataset(dfx: pd.DataFrame) -> pd.DataFrame:
    # 이미 split 컬럼이 있어도 덮어쓰도록 초기화
    dfx = dfx.copy()
    dfx[SPLIT_COL] = None

    # 너무 작은 데이터셋이면(예: 1~2개) valid가 비거나 깨질 수 있어 예외 처리
    n = len(dfx)
    if n < 5:
        # 최소한 1개는 valid로 보내고 싶으면 아래처럼 처리
        # 단, n=1이면 valid만 생김
        valid_n = max(1, int(round(n * (1 - TRAIN_RATIO))))
        valid_idx = dfx.sample(n=valid_n, random_state=SEED).index
        dfx.loc[valid_idx, SPLIT_COL] = "valid"
        dfx.loc[dfx[SPLIT_COL].isna(), SPLIT_COL] = "train"
        return dfx

    # stratify 가능 여부 체크 (각 라벨 최소 2개 이상 권장)
    vc = dfx[LABEL_COL].value_counts(dropna=False)
    can_stratify = (vc.min() >= 2) and (vc.shape[0] >= 2)

    test_size = 1 - TRAIN_RATIO

    if can_stratify:
        train_idx, valid_idx = train_test_split(
            dfx.index,
            test_size=test_size,
            random_state=SEED,
            shuffle=True,
            stratify=dfx[LABEL_COL]
        )
    else:
        train_idx, valid_idx = train_test_split(
            dfx.index,
            test_size=test_size,
            random_state=SEED,
            shuffle=True
        )

    dfx.loc[train_idx, SPLIT_COL] = "train"
    dfx.loc[valid_idx, SPLIT_COL] = "valid"
    return dfx


# dataset_id_col 결정(위에서 DATASET_ID_COL 따라 달라짐)
dataset_id_col = "dataset_id" if DATASET_ID_COL is None else DATASET_ID_COL

splitted = (
    all_df
    .groupby(dataset_id_col, group_keys=False)
    .apply(split_one_dataset)
    .reset_index(drop=True)
)

os.makedirs(OUT_DIR, exist_ok=True)

# 전체 통합 + split 포함
combined_path = Path(OUT_DIR) / "metadata_all_combined_with_split.csv"
splitted.to_csv(combined_path, index=False)

# train/valid 별로도 따로 저장
train_path = Path(OUT_DIR) / "metadata_train.csv"
valid_path = Path(OUT_DIR) / "metadata_valid.csv"

splitted[splitted[SPLIT_COL] == "train"].to_csv(train_path, index=False)
splitted[splitted[SPLIT_COL] == "valid"].to_csv(valid_path, index=False)

print("DONE")
print(f"- combined: {combined_path}")
print(f"- train   : {train_path}  (n={ (splitted[SPLIT_COL]=='train').sum() })")
print(f"- valid   : {valid_path}  (n={ (splitted[SPLIT_COL]=='valid').sum() })")
print("Per-dataset split counts:")
print(splitted.groupby([dataset_id_col, SPLIT_COL]).size().unstack(fill_value=0))
