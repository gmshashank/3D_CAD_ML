import os
import torch
import shutil
import urllib
from zipfile import ZipFile


class ModelNet(torch.utils.data.Dataset):
    classes = {
        "airplane": 0,
        "bathtub": 1,
        "bed": 2,
        "bench": 3,
        "bookshelf": 4,
        "bottle": 5,
        "bowl": 6,
        "car": 7,
        "chair": 8,
        "cone": 9,
        "cup": 10,
        "curtain": 11,
        "desk": 12,
        "door": 13,
        "dresser": 14,
        "flower_pot": 15,
        "glass_box": 16,
        "guitar": 17,
        "keyboard": 18,
        "lamp": 19,
        "laptop": 20,
        "mantel": 21,
        "monitor": 22,
        "night_stand": 23,
        "person": 24,
        "piano": 25,
        "plant": 26,
        "radio": 27,
        "range_hood": 28,
        "sink": 29,
        "sofa": 30,
        "stairs": 31,
        "stool": 32,
        "table": 33,
        "tent": 34,
        "toilet": 35,
        "tv_stand": 36,
        "vase": 37,
        "wardrobe": 38,
        "xbox": 39,
    }

    def __init__(self, root_dir="", transform=[], mode="train", classes=[]):
        self.root_dir = root_dir
        self.transform = transform
        self._download_data()

    def _download_data(self):
        if os.path.exists(self.root_dir):
            return

        print(f"ModelNet40 dataset does not exist in root directory{self.root_dir}.\n")
        print("Downloading ModelNet40 dataset.")

        os.makedirs(self.root_dir)
        url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        data = urllib.request.urlopen(url)
        filename = url.rpartition("/")[2][:-5]
        file_path = os.path.join(self.root_dir, filename)
        with open(file_path, mode="wb") as f:
            d = data.read()
            f.write(d)

        print("Extracting dataset.")
        with ZipFile(file_path, mode="wb") as zip_f:
            zip_f.extractall(self.root_dir)

        os.remove(file_path)
        extracted_dir = os.path.join(self.root_dir, filename[:-5])
        for d in os.listdir(extracted_dir):
            shutil.move(src=os.path.join(extracted_dir, d), dst=self.root_dir)
            shutil.rmtree(extracted_dir)
