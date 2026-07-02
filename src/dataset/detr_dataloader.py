import os

import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from src.utils.utils import load_config

def coco_collate_fn(batch):
    return tuple(zip(*batch))


def get_detr_transformer():
    return T.Compose([
        T.ToImage(),
        T.Resize((640, 640)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class NormalizedDetectionWrapper(torch.utils.data.Dataset):
    def __init__(self, coco_dataset, img_size=640):
        self.coco_dataset = coco_dataset
        self.img_size = float(img_size)

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        image, targets = self.coco_dataset[idx]
        
        boxes = []
        labels = []
        
        for target in targets:
            x, y, w, h = target['bbox']
            cx = x + w / 2.0
            cy = y + h / 2.0
            
            cx_norm = cx / self.img_size
            cy_norm = cy / self.img_size
            w_norm = w / self.img_size
            h_norm = h / self.img_size
            
            boxes.append([cx_norm, cy_norm, w_norm, h_norm])
            labels.append(target['category_id'] - 1) # backgtound is last class
            
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        detr_target = {
            "boxes": boxes,
            "labels": labels
        }
        
        return image, detr_target



def get_detr_dataloader(is_valid=False, is_test=False, 
                    epoch_id=None):
    
    cfg = load_config()

    data_cfg = cfg["data"]

    data_transforms = get_detr_transformer()

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

    img_size = data_cfg["img_size"]
    coco_dataset = NormalizedDetectionWrapper(coco_dataset, img_size=img_size)

    dataloader = DataLoader(
        coco_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle = not is_valid and not is_test,
        num_workers=2,
        collate_fn=coco_collate_fn
    )

    return dataloader

