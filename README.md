# Deepfake Detection (SwinV2 + EFFORT SVD-Residual Fine-tuning)

**HAI(하이)! - Hecto AI Challenge : 2025 하반기 헥토 채용 AI 경진대회** 참여를 위해 구현한  
Swin Transformer V2 기반 **딥페이크(Real/Fake) 이진 분류** 파이프라인입니다.

- **DDP + AMP** 기반 학습 지원
- 선택적으로 **EFFORT 스타일 SVD-Residual(저랭크 잔차 업데이트)**를 SwinV2 **마지막 stage**에 적용하여 **파라미터 효율적 미세튜닝**을 수행합니다.

---

## 1. 핵심 아이디어

### SwinV2 Large (입력 384)
- 모델: `swinv2_large_patch4_window12to24_192to384_22kto1k_ft`
- 입력 크기: `IMG_SIZE=384`
- Window size: `24`
- Stage depth: `[2, 2, 18, 2]`
- Num heads: `[6, 12, 24, 48]`
- 학습 설정 예: `EPOCHS=300`, `BASE_LR=2e-5`, `WEIGHT_DECAY=1e-8`

### EFFORT: SVD-Residual Linear 교체
일부 `nn.Linear` 레이어를 SVD 기반으로 분해하여 다음 형태로 치환합니다.

- 원래 가중치 `W`를 SVD로 분해한 뒤,
  - 큰 성분(top-r)으로 만든 `W_main`은 **고정(frozen)**
  - 나머지 성분을 `U_res, S_res, V_res`로 두고 **잔차(ΔW)만 학습(trainable)**

즉,

- `W ≈ W_main (frozen) + U_res @ diag(S_res) @ V_res (trainable)`

#### EFFORT 적용 위치(variant)
- `attn` : 마지막 stage의 **Attention(qkv, proj)** 만 SVDResidual로 교체
- `attn_mlp` : 마지막 stage의 **Attention(qkv, proj) + MLP(fc1, fc2)** 까지 교체

#### 정규화(학습 안정화) loss
학습 시(옵션) 아래 정규화를 추가합니다.

- `L_orth`: residual의 `U_res / V_res`가 서로 **겹치지 않도록(직교에 가깝게)** 유도
- `L_s`: `|S_residual|` 크기를 억제하는 규제(절댓값 평균)

`--lambda_orth`, `--lambda_s`가 0보다 크면 정규화가 `CE loss`에 더해집니다.

---

## 2. 환경 세팅

> 아래는 예시입니다. (CUDA/torch 버전은 실행 환경에 맞게 조정)

```bash
conda create -n deepfake_swin python=3.10 -y
conda activate deepfake_swin

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm opencv-python tqdm scikit-learn pandas pyyaml
```

---

## 3. 데이터 준비

### (Train/Val) 메타데이터 CSV
학습은 --train_csv, --val_csv를 사용합니다.


예)
```
/path/to/metadata_train.csv
/path/to/metadata_valid.csv
```

CSV 컬럼 구성은 data/build_loader_deepfake.py의 구현을 기준으로 맞춰야 합니다.

### (Infer) 테스트 폴더/샘플 제출 파일
추론 모드는 아래 규약을 그대로 사용합니다.

./test_data/ : 테스트 파일들이 위치
./sample_submission.csv : 대회 제공 샘플 제출 양식

결과는 ./output/baseline_submission.csv 로 저장

---

## 4. 학습 (Train)
### 기본 학습 (SwinV2 fine-tuning)

```
python main.py \
  --mode train \
  --cfg ./configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml \
  --data_path /path/to/dataset_root \
  --train_csv /path/to/metadata_train.csv \
  --val_csv /path/to/metadata_valid.csv \
  --batch_size 32 \
  --num_frames 5
```

### EFFORT(SVD-Residual) 켜기
--use_effort는 기본값이 True로 잡혀 있어, 실험 재현을 위해 명시하는 걸 권장합니다.

#### Attention만 교체:
```
python main.py \
  --mode train \
  --cfg ./configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml \
  --train_csv /path/to/metadata_train.csv \
  --val_csv /path/to/metadata_valid.csv \
  --data_path /path/to/dataset_root \
  --use_effort \
  --effort_variant attn \
  --k_residual 8 \
  --lambda_orth 1e-3 \
  --lambda_s 1e-4
```

#### Attention + MLP까지 교체:
```
python main.py \
  --mode train \
  --cfg ./configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml \
  --train_csv /path/to/metadata_train.csv \
  --val_csv /path/to/metadata_valid.csv \
  --data_path /path/to/dataset_root \
  --use_effort \
  --effort_variant attn_mlp \
  --k_residual 8 \
  --lambda_orth 1e-3 \
  --lambda_s 1e-4
```

---

## 5. 평가 지표
Validation에서 다음을 로깅합니다.

- Acc@1
- ROC-AUC (각 rank의 예측을 gather한 뒤 rank0에서 계산)

---

## 6. 추론(Inference) & 제출 파일 생성
```
python main.py \
  --mode infer \
  --cfg ./configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml \
  --ckpt /path/to/checkpoint.pth \
  --num_frames 5
```
- ./test_data 안의 파일들을 순회하며 전처리 후,
- 프레임별 fake 확률을 구해 평균을 내고,
- ./output/baseline_submission.csv로 저장합니다.

---

## 7. 폴더 구조(예시)
```
deepfake-detection-swinv2-effort/
├─ main.py
├─ configs/
│  └─ swinv2/
│     └─ swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml
├─ models/
│  ├─ SVD.py              # SVDResidualLinear + effort 적용 함수
│  └─ ...
├─ data/
│  ├─ build_loader_deepfake.py
│  ├─ preprocessing.py
│  └─ ...
├─ preprocessing/
├─ utils/
├─ test_data/             # (infer) 테스트 파일 위치
└─ output/                # (infer) 제출 csv 출력
```

---

## 8. Note
--k_residual은 residual 쪽에서 “학습시키는 특이값 성분 개수”를 의미합니다. 값이 클수록 표현력↑, 파라미터/불안정성↑ 가능.
--train_svd_bias로 SVDResidualLinear의 bias 학습 여부를 제어할 수 있습니다.
