import os
import json
import argparse

import cv2
import albumentations as A
from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.utils import load_config

def slice_image(
        image,
        boxes,
        category_ids,
        target_size=640,
        overlap=0.1
    ):
    height, width = image.shape[:2]
    step = int(target_size * (1 - overlap))

    sliced_results = []

    for y in range(0, height, step):
        for x in range(0, width, step):

            y_end = min(y + target_size, height)
            x_end = min(x + target_size, width)
            y_start = max(0, y_end - target_size)
            x_start = max(0, x_end - target_size)

            crop_image = image[y_start:y_end, x_start:x_end]
            crop_boxes = []
            crop_cats = []

            for box, category in zip(boxes, category_ids):
                bx, by, bw, bh = box

                bx_center = bx + bw / 2
                by_center = by + bh / 2

                if x_start <= bx_center <= x_end \
                and y_start <= by_center <= y_end:

                    nx = bx - x_start
                    ny = by - y_start
                    nw, nh = bw, bh

                    if nx < 0:
                        nw = nw + nx
                        nx = 0
                    if ny < 0:
                        nh = nh + ny
                        ny = 0

                    if nx + nw > target_size:
                        nw = target_size - nx
                    if ny + nh > target_size:
                        nh = target_size - ny

                    if nw > 3 and nh > 3:
                        crop_boxes.append([nx, ny, nw, nh])
                        crop_cats.append(category)

            sliced_results.append({
                "image": crop_image,
                "boxes": crop_boxes,
                "category_ids": crop_cats
            })

    return sliced_results


def save_new_image(
        new_coco,
        patch,
        img_data,
        img_global_id,
        ann_global_id,
        dest_dir,
        p_idx=None
    ):
    if p_idx is None:
        new_name = img_data["file_name"]
    else:
        name_no_extension = os.path.splitext(img_data["file_name"])[0]
        new_name = f"{name_no_extension}_p{p_idx}.jpg"

    cv2.imwrite(os.path.join(dest_dir, new_name), patch["image"])
    new_coco["images"].append({
        "id": img_global_id,
        "file_name": new_name,
        "width": patch["image"].shape[1],
        "height": patch["image"].shape[0]
    })

    for bbox, category in zip(patch["boxes"], patch["category_ids"]):
        new_coco["annotations"].append({
            "id": ann_global_id,
            "image_id": img_global_id,
            "category_id": category,
            "bbox": list(bbox),
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        ann_global_id += 1

    img_global_id += 1

    return new_coco, img_global_id, ann_global_id


def create_annotations_by_img(annotations):
    annotations_by_img = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)
    return annotations_by_img


def process_single_image_crop(
        img_data, 
        source_dir, 
        dest_dir, 
        ann_dict
    ):
    image_path = os.path.join(source_dir, img_data["file_name"])
    image = cv2.imread(image_path)

    img_anns = ann_dict.get(img_data["id"], [])
    boxes = [ann["bbox"] for ann in img_anns]
    category_ids = [ann["category_id"] for ann in img_anns]

    return slice_image(image, boxes, category_ids), img_data


def process_single_image_aug(
        img_data, 
        source_dir, 
        dest_dir, 
        ann_dict, 
        pipeline
    ):
    image_path = os.path.join(source_dir, img_data["file_name"])
    image = cv2.imread(image_path)

    img_anns = ann_dict.get(img_data["id"], [])
    boxes = [ann["bbox"] for ann in img_anns]
    category_ids = [ann["category_id"] for ann in img_anns]

    augmented = pipeline(image=image, 
                         bboxes=boxes, 
                         category_ids=category_ids)
    patch = {
        "image": augmented["image"],
        "boxes": augmented["bboxes"],
        "category_ids": augmented["category_ids"]
    }
    return patch, img_data


def process_images_in_dataset(
        source_dir,
        dest_dir,
        coco_data,
        num_workers=1
    ):
    new_coco = create_new_coco(coco_data)
    ann_dict = create_annotations_by_img(coco_data["annotations"])
    img_global_id, ann_global_id = 0, 0

    max_in_flight = num_workers * 2
    semaphore = Semaphore(max_in_flight)
 
    def worker(img_data):
        try:
            return process_single_image_crop(
                img_data, 
                source_dir, 
                dest_dir, 
                ann_dict
            )
        except Exception as e:
            print(f"Error with {img_data['file_name']}: {e}")
            return None
        finally:
            semaphore.release()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for img_data in coco_data["images"]:
            semaphore.acquire()
            future = executor.submit(worker, img_data)
            futures[future] = img_data
 
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            sliced_results, img_data = result
            for p_idx, patch in enumerate(sliced_results):
                new_coco, img_global_id, ann_global_id = save_new_image(
                    new_coco=new_coco,
                    patch=patch,
                    img_data=img_data,
                    img_global_id=img_global_id,
                    ann_global_id=ann_global_id,
                    dest_dir=dest_dir,
                    p_idx=p_idx
                )
 
    return new_coco

def get_augmented_pipeline():
    return A.Compose([
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625,
                           scale_limit=0.2,
                           rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT),
        A.HueSaturationValue(p=0.3,
                             hue_shift_limit=10,
                             sat_shift_limit=20,
                             val_shift_limit=30)
    ],
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["category_ids"],
        min_visibility=0.7
    ))


def process_augmentations(
        source_dir,
        dest_dir,
        coco_data,
        epoch_idx,
        num_workers=1
    ):
    pipeline = get_augmented_pipeline()
    new_coco = create_new_coco(coco_data)
    ann_dict = create_annotations_by_img(coco_data["annotations"])
    img_global_id, ann_global_id = 0, 0

    max_in_flight = num_workers * 4
    semaphore = Semaphore(max_in_flight)
 
    def worker(img_data):
        try:
            return process_single_image_aug(
                img_data, 
                source_dir, 
                dest_dir, 
                ann_dict, 
                pipeline
            )
        except Exception as e:
            print(f"Error in aug epoch {epoch_idx} " \
                  f"with {img_data['file_name']}: {e}")
            return None
        finally:
            semaphore.release()
 
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for img_data in coco_data["images"]:
            semaphore.acquire()
            future = executor.submit(worker, img_data)
            futures[future] = img_data
 
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            patch, img_data = result
            new_coco, img_global_id, ann_global_id = save_new_image(
                new_coco=new_coco,
                patch=patch,
                img_data=img_data,
                img_global_id=img_global_id,
                ann_global_id=ann_global_id,
                dest_dir=dest_dir
            )
 
    return new_coco
 

def create_new_coco(coco_data):
    return {
        "categories": coco_data["categories"],
        "images": [],
        "annotations": []
    }


def read_annotation_file(source_dir):
    annotation_file_path = os.path.join(
        source_dir, 
        "_annotations.coco.json"
    )
    with open(annotation_file_path, "r") as f:
        print(f"Reading {annotation_file_path}...")
        coco_data = json.load(f)
        print("Successfully read the file!")
    return coco_data


def write_new_annotation_file(dest_dir, coco_data):
    dest_annotation_file_path = os.path.join(
        dest_dir, 
        "_annotations.coco.json"
    )
    with open(dest_annotation_file_path, "w") as f:
        print(f"Writing {dest_annotation_file_path}...")
        json.dump(coco_data, f)
        print("Successfully wrote the file!")


def prepare_dataset_crop(raw_dir, processed_dir, num_workers=1):
    splits = ["valid", "test", "train"]

    for split in splits:
        print(f"Working with {split} folder")

        source_dir = os.path.join(raw_dir, split)
        dest_dir = os.path.join(processed_dir, split)

        if split == "train":
            dest_dir = os.path.join(dest_dir, "origin")

        os.makedirs(dest_dir, exist_ok=True)

        coco_data = read_annotation_file(source_dir)
        new_coco_data = process_images_in_dataset(
            source_dir=source_dir,
            dest_dir=dest_dir,
            coco_data=coco_data,
            num_workers=num_workers
        )
        write_new_annotation_file(dest_dir=dest_dir, 
                                  coco_data=new_coco_data)


def prepare_dataset_epochs(
        processed_dir, 
        start_ep, 
        end_ep, 
        num_workers=1
    ):
    source_dir = os.path.join(processed_dir, "train", "origin")
    coco_data = read_annotation_file(source_dir)

    for ep in range(start_ep, end_ep + 1):
        print(f"Augmenting epoch: {ep}...")
        dest_dir = os.path.join(processed_dir, "train", f"epoch_{ep}")
        os.makedirs(dest_dir, exist_ok=True)

        new_coco_data = process_augmentations(
            source_dir=source_dir,
            dest_dir=dest_dir,
            coco_data=coco_data,
            epoch_idx=ep,
            num_workers=num_workers
        )
        write_new_annotation_file(dest_dir=dest_dir, 
                                  coco_data=new_coco_data)


def main():

    cfg = load_config()
    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--epochs",
        type=str,
        default=None,
        help="Epochs as start:end or end (e.g. 3 or 2:5)"
    )
    parser.add_argument(
        "-m", "--multithread",
        type=int,
        default=1,
        metavar="WORKERS",
        help="Number of worker threads (default: 1)"
    )
    args = parser.parse_args()

    num_workers = max(1, args.multithread)

    if args.epochs:
        try:
            parts = list(map(int, args.epochs.split(":")))
            if len(parts) == 1:
                start_ep, end_ep = 1, parts[0]
            elif len(parts) == 2:
                start_ep, end_ep = parts
            else:
                raise ValueError

            prepare_dataset_epochs(processed_dir, start_ep, 
                                   end_ep, num_workers)

        except ValueError:
            print("Error: --epochs format must be end or start:end " \
                  "(e.g. 3 or 2:5)")
            return
    else:
        prepare_dataset_crop(raw_dir, processed_dir, num_workers)


if __name__ == "__main__":
    main()