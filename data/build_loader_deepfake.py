# data/build_loader_deepfake.py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from data.deepfake_dataset import DeepfakeDataset


def build_loader_deepfake(config, train_csv: str, val_csv: str, num_frames: int, data_path: str):
    img_size = int(config.DATA.IMG_SIZE)

    # (중요) 너의 inference 전처리(get_full_frame_padded)와 일관되게 "Resize/Pad 이후" 학습하도록 설계됨.
    # 여기서는 추가 증강만 가볍게.
    train_tf = transforms.Compose([
        # transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        # transforms.Resize(img_size),
        # transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    dataset_train = DeepfakeDataset(
        csv_path=train_csv,
        num_frames=num_frames,
        target_size=(img_size, img_size),
        mode="train",
        transform=train_tf,
        root_dir=data_path,
    )
    dataset_val = DeepfakeDataset(
        csv_path=val_csv,
        num_frames=num_frames,
        target_size=(img_size, img_size),
        mode="val",
        transform=val_tf,
        root_dir=data_path,
    )

    is_dist = dist.is_available() and dist.is_initialized()
    if is_dist:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        val_sampler = DistributedSampler(dataset_val, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    mixup_fn = None  # 원하면 나중에 timm Mixup 그대로 붙여도 됨
    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_infer_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),  # ★ 이 줄이 핵심
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])