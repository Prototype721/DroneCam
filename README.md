# VisDrone Object Detection — Multi-Model Comparison

This project trains and compares several object detection architectures on the
VisDrone dataset: Faster R-CNN, SSD, DETR (via torchvision/PyTorch) and
YOLOv8 / YOLO11 (via Ultralytics). It was built as a university project to
see how classic two-stage detectors, single-stage detectors, and
transformer-based detectors compare on drone imagery, which is a pretty
tough case because of small objects and high object density.

## Dataset

VisDrone (COCO-annotated version), split into `train` / `valid` / `test`.
The raw dataset is not included in this repo — it's expected under
`data/raw/` with the structure the preprocessing script expects
(`_annotations.coco.json` + images per split).

Because the raw images are large and full training on the whole dataset
takes a long time on my hardware, most of the runs in this repo were done
on a reduced subset (see `scripts/prepare_subset.py`) rather than the full
train/valid/test split. That's noted again in the Results section below.

## Preprocessing

To train models correctly we can't use resize due to low objects detection
in dataset, so instead image is sliced  into 640x640 patches with overlap.
The slicing images saved in `test`, `valid` and `train / origin` folders

For learning on weak device the augmentation porcess is done before the training
and all augmentated images are stored in `train / epoch_{i}` for each epoch.
Augmentation consists of:

1) A.HorizontalFlip(),
2) A.RandomBrightnessContrast(brightness_limit=0.2,
                            contrast_limit=0.2),
3) A.ShiftScaleRotate(shift_limit=0.0625,
                    scale_limit=0.2,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT),
4) A.HueSaturationValue(p=0.3,
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=30)


## Dataloaders

Two different dataloader/wrapper setups were needed because the model
families expect different target formats:

- `TorchvisionDetectionWrapper` (torch_dataloader.py) — converts raw COCO
  annotations into `xyxy` boxes + labels, the format Faster R-CNN and SSD
  expect from torchvision.
- `DETRDatasetWrapper` / `NormalizedDetectionWrapper` (detr_dataloader.py) —
  converts boxes into normalized `cxcywh` format, which is what DETR expects,
  and resizes everything to a fixed size so images in a batch can be
  stacked.
- YOLOv8/YOLO11 don't use these at all — Ultralytics reads directly from a
  `data.yaml` pointing at a YOLO-format folder (`images/`, `labels/`).

To do the last we need to also download dataset in pure yolo8 format and save 
in `data / yolo` folder.


## Project structure

```
configs/
  default.yaml          # all hyperparameters per model + data paths
src/
  utils/
    utils.py             # config loading, seeding
    logger.py             # TrainingLogger: JSON logs + matplotlib plots
  dataset/
    dataset.py            # raw -> processed preprocessing (slicing, augmentation)
    torch_dataloader.py    # dataloader for Faster R-CNN / SSD
    detr_dataloader.py     # dataloader for DETR
  evaluation/
    metrics.py             # mAP / precision / recall / F1 / mean IoU
  models/
    model_interface.py     # BaseModel abstract class
    faster_rcnn.py
    ssd.py
    detr.py
    yolo8.py
    yolo11.py
    factory.py              # ModelFactory: name -> model class
  training/
    train.py               # train_model(): orchestrates train + evaluate
main.py                    # CLI entry point

```

## How to run

Preprocess the full dataset (or see the subset option below):

```bash
python -m src.dataset.dataset                 # slice raw images
python -m src.dataset.dataset --epochs 5       # generate augmented epoch folders
```

Train a model:

```bash
python main.py --model faster_rcnn
python main.py --model ssd
python main.py --model detr
python main.py --model yolo8
python main.py --model yolo11
```

Evaluate only, using saved weights:

```bash
python main.py --model faster_rcnn --weights models_weights/faster_rcnn_last.pth --evaluate-only
```

Each model logs per-epoch loss + metrics to `results/logs/<model>_log.json`
and saves loss / metrics / per-class AP plots to `results/plots/`.



## Notes / limitations

- Trained on a reduced subset (not the full VisDrone train/valid/test split)
  due to hardware constraints, so numbers here aren't directly comparable to
  full-dataset benchmarks.
- DETR in particular tends to need many more epochs than 5 to reach
  reasonable mAP — the numbers for it in this run should be read as "does
  the pipeline work correctly", not "is DETR competitive here".
- `device` is set to `cpu` for all models in `configs/default.yaml`; switch
  to `cuda` if a GPU is available.