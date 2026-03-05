import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
import json
import random
import datetime

from torch.amp import autocast

from tqdm import tqdm
# from transformers import ViTForImageClassification, ViTImageProcessor

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data.build_loader_deepfake import build_loader_deepfake, build_infer_transform
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger

from data.preprocessing import *
from utils.utils import *
from models.SVD import *

### Swin ###
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg',
                        default='./configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml',
                        type=str, metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch_size', type=int, default=2, help="batch size for single GPU")
    parser.add_argument('--data_path', type=str,
                        default=r'D:\contest\2026_deepfake', help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained', default='./weights/swinv2_large_patch4_window12_192_22k.pth',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='./output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default='swin1', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # for pytorch >= 2.0, use `os.environ['LOCAL_RANK']` instead
    # (see https://pytorch.org/docs/stable/distributed.html#launch-utility)
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    parser.add_argument('--train_csv', type=str,
                        default=r'D:\contest\2026_deepfake\metadata_train.csv',
                        help='train csv (path,label)')
    parser.add_argument('--val_csv', type=str,
                        default=r'D:\contest\2026_deepfake\metadata_valid.csv',
                        help='val csv (path,label)')
    parser.add_argument('--num_frames', type=int, default=5, help='frames to sample from video')
    parser.add_argument('--mode', type=str, default='infer', choices=['train', 'infer'])
    parser.add_argument('--ckpt', type=str, help='(infer) checkpoint path .pth')
    ## effort
    parser.add_argument('--use_effort', action='store_true', default=True, help='apply effort-style SVD replacement')
    parser.add_argument('--effort_variant', type=str, default='attn_mlp', choices=['attn', 'attn_mlp'])
    parser.add_argument('--k_residual', type=int, default=8)
    parser.add_argument('--train_svd_bias', action='store_true')
    parser.add_argument('--lambda_orth', type=float, default=1e-3)
    parser.add_argument('--lambda_s', type=float, default=1e-4)
    parser.add_argument(
        '--test_dir',
        type=str,
        required=True,
        help='inference image directory (jpg only)'
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader_deepfake(
        config, args.train_csv, args.val_csv, args.num_frames, args.data_path
    )

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    model_without_ddp = model

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    def apply_effort_if_needed(m: nn.Module):
        if not getattr(args, "use_effort", False):
            return

        if args.effort_variant == "attn":
            apply_effort_swin_last_stage_attn_only(
                m,
                k_residual=args.k_residual,
                freeze_others=False,                 # 전체 FT
                train_svd_bias=args.train_svd_bias
            )
        else:
            apply_effort_swin_last_stage_attn_and_mlp(
                m,
                k_residual=args.k_residual,
                freeze_others=False,                 # 전체 FT
                train_svd_bias=args.train_svd_bias
            )

        # 전체 FT에서도 bias 학습 여부 강제
        set_svd_bias_trainable(m, args.train_svd_bias)

    if config.MODEL.RESUME:
        # effort 구조 먼저 만들어야 key가 맞음
        apply_effort_if_needed(model_without_ddp)
    else:
        # pretrained 로드
        if config.MODEL.PRETRAINED:
            load_pretrained(config, model_without_ddp, logger)

            # # pretrained 검증
            # acc1, loss = validate(config, data_loader_val, model_without_ddp)
            # logger.info(f"[Pretrained] Acc@1 {acc1:.2f}% | loss {loss:.4f}")

        # effort 적용
        apply_effort_if_needed(model_without_ddp)

    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    optimizer = build_optimizer(config, model_without_ddp)

    is_dist = dist.is_available() and dist.is_initialized()

    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model_without_ddp, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
        )
    else:
        model = model_without_ddp

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        # effort 구조는 이미 적용된 상태여야 함 (위에서 apply_effort_if_needed가 수행됨)
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, loss = validate(config, data_loader_val, model)
        logger.info(f"[Resume] Acc@1 {acc1:.2f}% | loss {loss:.4f}")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if hasattr(data_loader_train, "sampler") and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, args)
        rank0 = get_rank()
        if rank0 == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)

        acc1, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, args):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        else:
            targets = targets.long()

        with autocast("cuda", enabled=config.AMP_ENABLE):
            outputs = model(samples)

        ce_loss = criterion(outputs, targets)

        ## L_orth + |S_residual| 추가
        total_loss = ce_loss
        if args.use_effort and (args.lambda_orth > 0 or args.lambda_s > 0):
            base_model = model.module if hasattr(model, "module") else model
            reg, loss_orth, loss_s = compute_effort_regularizers(
                base_model,
                lambda_orth=args.lambda_orth,
                lambda_s=args.lambda_s
            )
            total_loss = total_loss + reg

        loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with autocast("cuda", enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1 = accuracy(output, target, topk=(1,))[0]
        acc1 = reduce_tensor(acc1)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')
    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


# inference
'''
def infer_fake_probs(pil_images: List[Image.Image]) -> List[float]:
    if not pil_images:
        return []

    probs: List[float] = []

    with torch.inference_mode():
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}
        logits = model(**inputs).logits
        batch_probs = F.softmax(logits, dim=1)[:, 1]
        probs.extend(batch_probs.cpu().tolist())

    return probs
'''

# Swin 용
# def infer_fake_probs(pil_images: List[Image.Image], model, device, tfm) -> List[float]:
#     if not pil_images:
#         return []
#
#     with torch.inference_mode():
#         x = torch.stack([tfm(im) for im in pil_images], dim=0).to(device, non_blocking=True)  # [B,3,H,W]
#         logits = model(x)  # [B,2]
#         probs = F.softmax(logits, dim=1)[:, 1]  # fake prob
#         return probs.detach().cpu().tolist()
def parse_video_id_and_frame(p: Path):
    """
    TEST_003_4.jpg  -> ('TEST_003', True)
    TEST_002.jpg    -> ('TEST_002', False)
    TEST_010.jpg    -> ('TEST_010', False)
    """
    stem = p.stem
    parts = stem.split("_")

    # TEST_xxx_frameIndex.jpg (underscore 2개 이상)
    if len(parts) >= 3 and parts[-1].isdigit():
        return "_".join(parts[:-1]), True
    else:
        return stem, False


# def infer_fake_probs(pil_images, model, device, tfm):
#     probs = []
#     with torch.inference_mode():
#         for im in pil_images:
#             x = tfm(im).unsqueeze(0).to(device)
#             logits = model(x)
#             prob = F.softmax(logits, dim=1)[0, 1]
#             probs.append(prob.item())
#     return probs

def infer_fake_probs(pil_images, model, device, tfm):
    probs = []

    with torch.inference_mode():
        for im in pil_images:
            x = tfm(im).unsqueeze(0).to(device)  # [1,3,H,W]
            logits = model(x)                    # forward 1장
            prob = F.softmax(logits, dim=1)[0, 1]
            probs.append(prob.item())

    return probs

if __name__ == '__main__':
    args, config = parse_option()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    if args.mode == "train":
        assert args.train_csv and args.val_csv, "--mode train 에서는 --train_csv, --val_csv가 필요해."

        is_dist_launch = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if is_dist_launch:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://",
                                    world_size=world_size, rank=rank)
            dist.barrier()
        else:
            rank = 0
            world_size = 1
            # 단일 GPU면 보통 cuda:0
            torch.cuda.set_device(0)

        seed = config.SEED + get_rank()
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        np.random.seed(seed); random.seed(seed)
        cudnn.benchmark = True

        '''
        # LR linear scaling (원본 유지)
        denom = 512.0

        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * world_size / denom
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * world_size / denom
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * world_size / denom
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS

        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()
        '''

        os.makedirs(config.OUTPUT, exist_ok=True)

        dist_rank = get_rank()
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist_rank, name=f"{config.MODEL.NAME}")

        if dist_rank == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")

        logger.info(config.dump())
        logger.info(json.dumps(vars(args)))

        # 학습 시작
        main(config, args)

    else:

        assert args.ckpt, "--mode infer 에서는 --ckpt가 필요해."

        assert args.test_dir, "--test_dir 필요"

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        model = build_model(config).to(DEVICE)

        model.eval()

        # effort 구조

        if args.use_effort:

            if args.effort_variant == "attn":

                apply_effort_swin_last_stage_attn_only(

                    model,

                    k_residual=args.k_residual,

                    freeze_others=False,

                    train_svd_bias=args.train_svd_bias

                )

            else:

                apply_effort_swin_last_stage_attn_and_mlp(

                    model,

                    k_residual=args.k_residual,

                    freeze_others=False,

                    train_svd_bias=args.train_svd_bias

                )

            try:

                set_svd_bias_trainable(model, args.train_svd_bias)

            except NameError:

                pass

        ckpt = torch.load(args.ckpt, map_location="cpu")

        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        model.load_state_dict(state, strict=True)

        img_size = int(config.DATA.IMG_SIZE)

        tfm = build_infer_transform(img_size)

        TEST_DIR = Path(args.test_dir)

        OUTPUT_DIR = Path("./output")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        test_name = TEST_DIR.name

        OUT_CSV = OUTPUT_DIR / f"submission_{test_name}_69ep.csv"

        # =========================

        # 1. sample_submission 로드 (확장자 유지용)

        # =========================

        submission = pd.read_csv(r"D:\contest\2026_deepfake\test_data\sample_submission.csv")

        filename_to_ext = {

            Path(f).stem: Path(f).suffix

            for f in submission["filename"]

        }

        # =========================

        # 2. 이미지 grouping

        # =========================

        video_frames = {}  # video_id -> [Path]

        for p in sorted(TEST_DIR.iterdir()):

            if not p.is_file():
                continue

            if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            video_id, _ = parse_video_id_and_frame(p)

            video_frames.setdefault(video_id, []).append(p)

        # =========================

        # 3. inference

        # =========================

        results = {}

        for video_id, frame_paths in tqdm(video_frames.items(), desc="Inference"):

            imgs = []

            for fp in frame_paths:

                try:

                    imgs.append(Image.open(fp).convert("RGB"))

                except Exception:

                    continue

            if not imgs:

                prob = 0.0

            else:

                probs = infer_fake_probs(imgs, model, DEVICE, tfm)

                prob = float(np.mean(probs)) if probs else 0.0

            # 🔥 원래 확장자 복원

            ext = filename_to_ext.get(video_id, ".mp4")

            results[f"{video_id}{ext}"] = prob

        # =========================

        # 4. CSV 저장

        # =========================

        submission["prob"] = submission["filename"].map(results).fillna(0.0)

        submission.to_csv(OUT_CSV, encoding="utf-8-sig", index=False)

        print(f"Saved submission to: {OUT_CSV}")


