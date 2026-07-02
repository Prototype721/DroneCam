"""
scripts/prepare_subset.py

Builds a small subset of the VisDrone dataset for fast local iteration on a
weak machine, then runs the project's OWN preprocessing pipeline
(src/dataset/dataset.py) on that subset — so the resulting processed/ and
yolo/ folders go through the exact same slicing/augmentation/conversion
logic a full run would use, just with far fewer images.

Steps:
  1. Sample N images (+ their annotations) per split from the raw VisDrone
     COCO folder. Original raw data is left untouched.
  2. Write the sample into data/raw_subset/<split>/ in the layout
     dataset.py expects (images + _annotations.coco.json).
  3. Call the project's real prepare_dataset_crop() on the subset
     -> data/processed/{train/origin,valid,test}
  4. Call the project's real prepare_dataset_epochs() for epochs 1..N
     -> data/processed/train/epoch_1 .. epoch_N   (used by faster_rcnn/ssd/detr)
  5. Convert the same subset into YOLO format
     -> data/yolo/images/{train,valid,test}
     -> data/yolo/labels/{train,valid,test}
     -> data/yolo/data.yaml               (used by yolo8/yolo11)

Run from the project root:
    python scripts/prepare_subset.py --n-train 200 --n-valid 40 --n-test 40 --epochs 5
"""

import os
import sys
import json
import random
import shutil
import argparse

# Allow "from src...." imports when run as `python scripts/prepare_subset.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.utils import load_config
from src.dataset.dataset import prepare_dataset_crop, prepare_dataset_epochs


# --------------------------------------------------------------------------
# Step 1-2: sample raw images/annotations into a subset raw folder
# --------------------------------------------------------------------------

def read_coco(split_dir):
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(ann_path, "r") as f:
        return json.load(f)


def sample_split(raw_dir, split, n_images, seed):
    source_dir = os.path.join(raw_dir, split)
    coco = read_coco(source_dir)

    images = coco["images"]
    n = min(n_images, len(images))

    rng = random.Random(seed)
    sampled_images = rng.sample(images, n)
    sampled_ids = {img["id"] for img in sampled_images}

    sampled_anns = [a for a in coco["annotations"] if a["image_id"] in sampled_ids]

    print(f"[{split}] sampled {len(sampled_images)}/{len(images)} images, "
          f"{len(sampled_anns)} annotations")

    return sampled_images, sampled_anns, coco["categories"], source_dir


def write_subset_split(dest_raw_dir, split, images, annotations, categories, source_dir):
    dest_dir = os.path.join(dest_raw_dir, split)
    os.makedirs(dest_dir, exist_ok=True)

    for img in images:
        src_path = os.path.join(source_dir, img["file_name"])
        dst_path = os.path.join(dest_dir, img["file_name"])
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    coco_out = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(os.path.join(dest_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco_out, f)

    return dest_dir


# --------------------------------------------------------------------------
# Step 5: convert the subset into YOLO format
# --------------------------------------------------------------------------

def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def build_yolo_split(subset_raw_dir, yolo_dir, split, yolo_split_name, cat_id_to_idx):
    split_dir = os.path.join(subset_raw_dir, split)
    coco = read_coco(split_dir)

    img_out_dir = os.path.join(yolo_dir, "images", yolo_split_name)
    lbl_out_dir = os.path.join(yolo_dir, "labels", yolo_split_name)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for img in coco["images"]:
        src_path = os.path.join(split_dir, img["file_name"])
        dst_path = os.path.join(img_out_dir, img["file_name"])
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

        stem = os.path.splitext(img["file_name"])[0]
        label_path = os.path.join(lbl_out_dir, f"{stem}.txt")

        lines = []
        for ann in anns_by_img.get(img["id"], []):
            cx, cy, nw, nh = coco_bbox_to_yolo(
                ann["bbox"], img["width"], img["height"]
            )
            cls_idx = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(f"[yolo/{yolo_split_name}] wrote {len(coco['images'])} images/labels")
    return coco["categories"]


def write_data_yaml(yolo_dir, categories, cat_id_to_idx):
    names_by_idx = {idx: cat["name"] for cat in categories
                     for idx in [cat_id_to_idx[cat["id"]]]}
    names_list = [names_by_idx[i] for i in range(len(names_by_idx))]

    yaml_lines = [
        f"path: {os.path.abspath(yolo_dir)}",
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        f"nc: {len(names_list)}",
        f"names: {names_list}",
    ]
    with open(os.path.join(yolo_dir, "data.yaml"), "w") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"[yolo] data.yaml written -> {os.path.join(yolo_dir, 'data.yaml')}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a small VisDrone subset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-valid", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=5,
                         help="How many augmented train/epoch_N folders to "
                              "generate for faster_rcnn/ssd/detr (default 5, "
                              "matches their config epoch counts).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-raw-dir", type=str, default="datta/raw")
    parser.add_argument("--yolo-dir", type=str, default="datta/yolo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = "datta/raw/visDrone.v5-visdrone_rcnn.coco"
    processed_dir = cfg["data"]["processed_dir"]

    n_per_split = {"train": args.n_train, "valid": args.n_valid, "test": args.n_test}

    # ---- 1-2: sample + write subset raw folder ----
    print("\n=== Sampling raw VisDrone data ===")
    all_categories = None
    for split, n in n_per_split.items():
        images, anns, categories, source_dir = sample_split(
            raw_dir, split, n, seed=args.seed
        )
        write_subset_split(
            args.subset_raw_dir, split, images, anns, categories, source_dir
        )
        all_categories = categories  # same across splits

    # ---- 3-4: run the project's real preprocessing on the subset ----
    print("\n=== Running project preprocessing (slicing) ===")
    prepare_dataset_crop(
        raw_dir=args.subset_raw_dir,
        processed_dir=processed_dir,
        num_workers=args.workers,
    )

    print("\n=== Running project preprocessing (epoch augmentations) ===")
    prepare_dataset_epochs(
        processed_dir=processed_dir,
        start_ep=1,
        end_ep=args.epochs,
        num_workers=args.workers,
    )

    # ---- 5: build YOLO-format folder from the same subset ----
    print("\n=== Building YOLO-format dataset ===")
    cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(
        sorted(all_categories, key=lambda c: c["id"])
    )}

    split_name_map = {"train": "train", "valid": "valid", "test": "test"}
    for split, yolo_split_name in split_name_map.items():
        build_yolo_split(
            args.subset_raw_dir, args.yolo_dir, split, yolo_split_name, cat_id_to_idx
        )

    write_data_yaml(args.yolo_dir, all_categories, cat_id_to_idx)

    print("\nDone. You now have:")
    print(f"  {processed_dir}/  (train/origin, train/epoch_1..{args.epochs}, valid, test)")
    print(f"  {args.yolo_dir}/  (images/, labels/, data.yaml)")
    print("\nYou can now run e.g.:")
    print("  python main.py --model faster_rcnn")
    print("  python main.py --model yolo8")


if __name__ == "__main__":
    main()