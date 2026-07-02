import os

import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from src.utils.utils import load_config

def coco_collate_fn(batch):
    return tuple(zip(*batch))


class TorchvisionDetectionWrapper(torch.utils.data.Dataset):

    def __init__(self, coco_dataset):
        self.coco_dataset = coco_dataset

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        image, targets = self.coco_dataset[idx]

        boxes = []
        labels = []

        for target in targets:
            x, y, w, h = target["bbox"]
            if w <= 0 or h <= 0:
                continue

            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])

            cat_id = target["category_id"]
            labels.append(cat_id)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}


def get_torch_dataloader(is_valid=False, is_test=False, 
                    epoch_id=None):
    
    cfg = load_config()

    data_cfg = cfg["data"]

    data_transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])

    path_dir = data_cfg["processed_dir"]

    if is_valid:
        split_path = data_cfg["valid_split"]
    elif is_test:
        split_path = data_cfg["test_split"]
    else:
        split_path = data_cfg["train_split"]
        if epoch_id is not None:
            split_path = os.path.join(split_path, f"epoch_{epoch_id}")
        else:
            split_path = os.path.join(split_path, f"origin")

    path_dir = os.path.join(path_dir, split_path)
    
    ann_file = os.path.join(path_dir, "_annotations.coco.json")

    coco_dataset = CocoDetection(
        root=path_dir,
        annFile=ann_file,
        transform=data_transforms
    )

    coco_dataset = TorchvisionDetectionWrapper(coco_dataset)

    dataloader = DataLoader(
        coco_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle = not is_valid and not is_test,
        num_workers=2,
        collate_fn=coco_collate_fn
    )

    return dataloader

