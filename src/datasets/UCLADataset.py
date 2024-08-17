import os
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from src.datasets.augment import ResizeSequence, RandomRotate
from src.datasets.utils import get_angular_motion

from src.graph.ntu_graph import Graph
from src.datasets.utils import get_angular_motion

from tqdm import tqdm


class NTUDataset(Dataset):
    train_subjects_file = "../../resources/train_subjects.txt"
    train_cameras_file = "../../resources/train_cameras.txt"

    def __init__(
        self,
        data_path,
        extra_data_path="",
        mode="train",
        split="x-subject",
        length_t=64,
        features="j",
        center=20,
        p_interval=[1],
        load_to_ram=False,
    ):
        super().__init__()
        self.graph = Graph()

        self.transform = transforms.Compose(
            [ResizeSequence(length_t, p_interval)]
        )
        self.augment = transforms.Compose([RandomRotate(0.3)])
        self.features = features
        self.samples = []
        self.labels = []
        self.mode = mode
        self.split = split
        self.load_to_ram = load_to_ram
        if self.mode not in ["train", "valid"]:
            raise NameError(f"Mode {self.mode} is invalid")
        self.train_ids = []
        if split == "x-subject":
            self.train_ids = []
            with open(
                os.path.join(
                    os.path.dirname(__file__), self.train_subjects_file
                ),
                "r",
            ) as f:
                lines = f.readlines()
                for line in lines:
                    self.train_ids.append(int(line))
        elif split == "x-setup":
            self.train_ids = [i for i in range(2, 33, 2)]
        elif split == "x-view":
            self.train_ids = []
            with open(
                os.path.join(
                    os.path.dirname(__file__), self.train_cameras_file
                ),
                "r",
            ) as f:
                lines = f.readlines()
                for line in lines:
                    self.train_ids.append(int(line))
        else:
            raise NameError(f"Split {split} is invalid")

        self.__read_data(data_path)
        if len(extra_data_path) > 0:
            self.__read_data(extra_data_path)

    def __read_data(self, path: str):
        print("-" * os.get_terminal_size().columns)
        print(f"Read {self.mode} data from {path}")
        for file in tqdm(
            sorted(os.scandir(path), key=lambda x: x.name),
            desc=f"Process samples",
            ncols=0,
        ):
            if self.split == "x-subject":
                c = "P"
            elif self.split == "x-setup":
                c = "S"
            else:
                c = "C"
            id = int(file.name[file.name.find(c) + 1 : file.name.find(c) + 4])
            label = (
                int(
                    file.name[file.name.find("A") + 1 : file.name.find("A") + 4]
                )
                - 1
            )
            if id in self.train_ids and self.mode == "train":
                if self.load_to_ram:
                    self.samples.append(torch.tensor(np.load(file.path)))
                else:
                    self.samples.append(file.path)

                self.labels.append(label)

            if id not in self.train_ids and self.mode == "valid":
                if self.load_to_ram:
                    self.samples.append(torch.tensor(np.load(file.path)))
                else:
                    self.samples.append(file.path)

                self.labels.append(label)
        print("-" * os.get_terminal_size().columns)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label = self.labels[index]

        if self.load_to_ram:
            sample = self.samples[index]
        else:
            sample_path = self.samples[index]

            sample = np.load(sample_path, allow_pickle=True)
            sample = torch.from_numpy(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.mode == "train" and self.augment is not None:
            sample = self.augment(sample)

        features = self.features.split(",")
        for id, f in enumerate(features):
            features[id] = f.strip()

        data = None

        for f in features:
            if f == "j":
                data = torch.cat([data, sample]) if data is not None else sample
            elif f == "b":
                data = (
                    torch.cat([data, self.__get_bones(sample)])
                    if data is not None
                    else self.__get_bones(sample)
                )
            elif f == "jm":
                data = (
                    torch.cat([data, self.__get_motion(sample)])
                    if data is not None
                    else self.__get_motion(sample)
                )
            elif f == "bm":
                data = (
                    torch.cat([data, self.__get_bones_motion(sample)])
                    if data is not None
                    else self.__get_bones_motion(sample)
                )
            elif f == "am":
                data = (
                    torch.cat([data, self.__get_angular_motion(sample)])
                    if data is not None
                    else self.__get_angular_motion(sample)
                )
            else:
                raise ValueError(f"Feature {f} is invalid")

        return data, label

    def __get_motion(self, sample: torch.Tensor):
        C, T, V, M = sample.size()
        diff = sample[:, 1:, :, :] - sample[:, :-1, :, :]

        joints_motion = torch.zeros_like(sample)
        joints_motion[:, 1:, :, :] = diff

        return joints_motion

    def __get_bones(self, sample: torch.Tensor):
        C, T, V, M = sample.size()
        bones = torch.zeros_like(sample)

        inward_A = self.graph.get_adjacency_matrix()[1]
        num_joints = inward_A.shape[0]
        for u in range(num_joints):
            for v in range(num_joints):
                if inward_A[u, v] == 0:
                    continue
                bones[:, :, u, :] = sample[:, :, v, :] - sample[:, :, u, :]
        return bones

    def __get_bones_motion(self, sample: torch.Tensor):
        C, T, V, M = sample.size()
        bones = self.__get_bones(sample)

        bones_motion = self.__get_motion(bones)

        return bones_motion

    def __get_angular_motion(self, sample: torch.Tensor):
        return get_angular_motion(sample, 20)
