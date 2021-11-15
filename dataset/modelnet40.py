import glob
import h5py
import numpy as np
import os
from numpy.lib.index_tricks import AxisConcatenator
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

    def __init__(
        self,
        train: bool = True,
        num_points: int = 1024,
        download: bool = True,
        transform=[],
        use_normals: bool = False,
    ):
        super().__init__()

        self.num_points = num_points
        self.transform = transform

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, os.pardir, "data")

        if download:
            self._download_data()
        self.data, self.labels = self._load_data(train, use_normals)
        if not train:
            self.shapes = self._read_class_ModelNet40()

    def __getitem__(self, idx: int):
        current_points = self.data[idx].copy()
        current_points = torch.from_numpy(current_points[: self.num_points, :]).float()
        if self.transform is not None:
            current_points = self.transform(current_points)
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        return current_points, label

    def __len__(self):
        return self.data.shape[0]

    def _download_data(self):
        if not os.path.exists(self.data_dir):
            print(f"ModelNet40 dataset does not exist in root directory{self.data_dir}.\n")
            os.makedirs(self.data_dir)

        if not os.path.exists(os.path.join(self.data_dir, "modelnet40_ply_hdf5_2048")):
            print("Downloading ModelNet40 dataset.")

            modelnet40_url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
            zip_file = os.path.basename(modelnet40_url)
            os.system(f"wget {modelnet40_url}; unzip {zip_file}")
            os.system(f"mv {zip_file[:-4]} {self.data_dir}")
            os.system(f"rm {zip_file}")

    def _load_data(self, train: bool, use_normals: bool):
        if train:
            partition = "train"
        else:
            partition = "test"
        all_data = []
        all_labels = []
        for h5_name in glob.glob(os.path.join(self.data_dir, "modelnet40_ply_hdf5_2048", f"ply_data_{partition}*.h5")):
            f = h5py.File(h5_name)
            if use_normals:
                data = np.concatenate([f["data"][:], f["normal"][:]], axis=-1).astype("float32")
            # if use_normals:data=np.concatenate([f["data"][:],f["normal"][:]],axis=1]).astype("float32")
            else:
                data = f["data"][:].astype("float32")
            label = f["label"][:].astype("int64")
            f.close()
            all_data.append(data)
            all_labels.append(label)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    def _read_class_ModelNet40(self):
        file = open(os.path.join(self.data_dir, "modelnet40_ply_hdf5_2048", "shape_names.txt"), "r",)
        shape_names = file.read()
        shape_names = np.array(shape_names.split("\n")[:-1])
        file.close()
        return shape_names

    def get_shape(self, label):
        return self.shapes[label]
