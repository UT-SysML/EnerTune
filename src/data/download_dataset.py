import fiftyone as fo
import fiftyone.zoo as foz
import os
import urllib.request
import zipfile
from pycocotools.coco import COCO
import requests
import kagglehub
from datasets import load_dataset
import torch
import pandas as pd
import numpy as np


# # Retinent Source: https://docs.voxel51.com/tutorials/open_images.html
# dataset = foz.load_zoo_dataset(
#         "open-images-v7",
#         split="test",
#         label_types=["detections"],
#         max_samples=64,
#         seed=51,
#         shuffle=True
#     )

# # Vit ImageNet:
# url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
# save_path = "tiny-imagenet-200.zip"

# if not os.path.exists(save_path):
#     print("Downloading Tiny-ImageNet...")
#     urllib.request.urlretrieve(url, save_path)

# extract_folder = "tiny-imagenet-200"
# if not os.path.exists(extract_folder):
#     print("Extracting dataset...")
#     with zipfile.ZipFile(save_path, "r") as zip_ref:
#         zip_ref.extractall("./")

# print("Tiny-ImageNet is ready!")


# Coco Dataset
# data_dir = "coco"
# data_type = "val2017"
# ann_file = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# if not os.path.exists(f"{data_dir}/annotations_trainval2017.zip"):
#     os.system(f"wget {ann_file} -P {data_dir}")
#     os.system(f"unzip {data_dir}/annotations_trainval2017.zip -d {data_dir}")
# coco = COCO(f"{data_dir}/annotations/instances_{data_type}.json")
# cat_ids = coco.getCatIds(catNms=["person"])
# img_ids = coco.getImgIds(catIds=cat_ids)
# N = 100
# images = coco.loadImgs(img_ids[:N])
# os.makedirs(f"{data_dir}/images", exist_ok=True)
# for img in images:
#     img_url = img["coco_url"]
#     img_data = requests.get(img_url).content
#     with open(f"{data_dir}/images/{img['file_name']}", "wb") as f:
#         f.write(img_data)
#     print(f"Downloaded: {img['file_name']}")

# IAM Dataset
# path = kagglehub.dataset_download("changheonkim/iam-trocr")
# print("Path to dataset files:", path)

# dataset = load_dataset("cnn_dailymail", "3.0.0")
# texts = dataset["test"]["article"][:40]  # Get 10 test articles
# for idx, text in enumerate(texts):
#     with open(f"./summarization/text_{idx}.txt", "w", encoding="utf-8") as f:
#         f.write(text)
    

timestamps = pd.date_range(start="2024-01-01", periods=100, freq="D")
values = np.cumsum(np.random.randn(100) * 5 + 50)  # Simulating an increasing trend

# Convert to torch tensor (1D format required by Chronos)
time_series_tensor = torch.tensor(values, dtype=torch.float32)

# Save to file (for reuse in the model)
torch.save(time_series_tensor, "./timeseries/historical_data.pt")