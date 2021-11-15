import cv2
import glob
import h5py
import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class ShapeNetPart(Dataset):
    def __init__(self, num_points: int = 1024, partition: str = "train", class_choice=None):
        self.data, self.label, self.seg = self._load_data(partition)
        self.cat_id = {
            "airplane": 0,
            "bag": 1,
            "cap": 2,
            "car": 3,
            "chair": 4,
            "earphone": 5,
            "guitar": 6,
            "knife": 7,
            "lamp": 8,
            "laptop": 9,
            "motor": 10,
            "mug": 11,
            "pistol": 12,
            "rocket": 13,
            "skateboard": 14,
            "table": 15,
        }
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice
        self.partseg_colors = self._load_color_partseg()

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item: int):
        point_cloud = self.data[item][: self.num_points]
        label = self.label[item]
        seg = self.seg[item][: self.num_points]
        if self.partition == "trainval":
            indices = list(range(point_cloud.shape[0]))
            np.random.shuffle(indices)
            point_cloud = point_cloud[indices]
            seg = seg[indices]
        return point_cloud, label, seg

    def __len__(self):
        return self.data.shape[0]

    def _download_data(self):
        data_dir = self.data_dir

        url = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(os.path.join(data_dir, "shapenet_part_seg_hdf5_data")):
            zip_file = os.path.basename(url)
            os.system(f"wget {url} --no-check-certificate; unzip {zip_file}")
            os.system(f"mv hdf5_data { os.path.join(data_dir,'shapenet_part_seg_hdf5_data')}")
            os.system(f"rm {zip_file}")

    def _load_data(self, partition: str):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        self._download_data()

        all_data = []
        all_label = []
        all_seg = []
        if partition == "trainval":
            files = glob.glob(os.path.join(self.data_dir, "shapenet_part_seg_hdf5_data", "*train*.h5")) + glob.glob(
                os.path.join(self.data_dir, "shapenet_part_seg_hdf5_data", "*val*.h5")
            )

        else:
            files = glob.glob(os.path.join(self.data_dir, "shapenet_part_seg_hdf5_data", "*partition*.h5"))

        for file_name in files:
            file = h5py.File(file_name, "r+")
            data = file["data"][:].astype("float32")
            label = file["label"][:].astype("int64")
            seg = file["pid"][:].astype("int64")
            file.close()
            all_data.append(data)
            all_label.append(label)
            all_seg.append(seg)

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        all_seg = np.concatenate(all_seg, axis=0)

        return all_data, all_label, all_seg

    def load_color_partseg(self):
        colors = []
        labels = []
        f = open("prepare_data/meta/partseg_colors.txt")
        for line in json.load(f):
            colors.append(line["color"])
            labels.append(line["label"])
        partseg_colors = np.array(colors)
        partseg_colors = partseg_colors[:, [2, 1, 0]]
        partseg_labels = np.array(labels)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_size = 1350
        img = np.zeros((1350, 1890, 3), dtype="uint8")
        cv2.rectangle(img, (0, 0), (1900, 1900), [255, 255, 255], thickness=-1)
        column_numbers = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        column_gaps = [320, 320, 300, 300, 285, 285]
        color_size = 64
        color_index = 0
        label_index = 0
        row_index = 16
        for row in range(0, img_size):
            column_index = 32
            for column in range(0, img_size):
                color = partseg_colors[color_index]
                label = partseg_labels[label_index]
                length = len(str(label))
                cv2.rectangle(
                    img,
                    (column_index, row_index),
                    (column_index + color_size, row_index + color_size),
                    color=(int(color[0]), int(color[1]), int(color[2])),
                    thickness=-1,
                )
                img = cv2.putText(
                    img,
                    label,
                    (column_index + int(color_size * 1.15), row_index + int(color_size / 2),),
                    font,
                    0.76,
                    (0, 0, 0),
                    2,
                )
                column_index = column_index + column_gaps[column]
                color_index = color_index + 1
                label_index = label_index + 1
                if color_index >= 50:
                    cv2.imwrite(
                        "prepare_data/meta/partseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0],
                    )
                    return np.array(colors)
                elif column + 1 >= column_numbers[row]:
                    break
            row_index = row_index + int(color_size * 1.3)
            if row_index >= img_size:
                break
