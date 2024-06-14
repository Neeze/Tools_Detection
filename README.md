# Project Title

## Description

```bash

Tools_Detection
├── config
├── data
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── val
│       ├── images
│       └── labels
├── scripts
└── src
    ├── models
    └── utils

```


## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed the version of `python==3.10.13`
* You have a `<Windows/Linux>` machine.
* Modules: numpy, torch, torchvision, tqdm, torchsummary, torchmetrics, torchinfo, sklearn, wandb, pycocotools, opencv-python, matplotlib, seaborn



## Quickstart

Clone the repo
```bash
git clone https://github.com/Neeze/Tools_Detection.git
```


Install Dependency
```bash
pip install numpy torch torchvision tqdm torchsummary torchmetrics torchinfo scikit-learn wandb pycocotools opencv-python matplotlib seaborn roboflow --quiet
```


Download dataset
```python
from roboflow import Roboflow
rf = Roboflow(api_key="WAWzuRQeZQxeCnJr4hHR")
project = rf.workspace("fpt-ycwb3").project("tool_detections")
version = project.version(1)
dataset = version.download("yolov8")
```

Training model

```bash
python scripts/yolo2voc.py --config config/faster_rcnn18_config.yaml
python scripts/train.py --config config/faster_rcnn18_config.yaml
```


