import os
import random
import math
import json
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from src.datasets.augment import ResizeSequence, RandomRotate
from src.datasets.utils import get_angular_motion

from src.graph.ucla_graph import Graph
from src.datasets.utils import get_angular_motion

from tqdm import tqdm


class UCLADataset(Dataset):
    train_subjects_file = "../../resources/train_subjects.txt"
    train_cameras_file = "../../resources/train_cameras.txt"

    def __init__(
        self,
        data_path,
        mode,
        features,
        repeat=5,
        random_choose=True,
        random_shift=False,
        random_move=False,
        window_size=52,
        normalization=False,
        debug=False,
        use_mmap=True,
    ):
        self.graph = Graph()
        self.features = features

        if "valid" in mode:
            self.mode = "valid"
            with open(
                os.path.join(
                    os.path.dirname(__file__), "../../resources/ucla_valid.json"
                )
            ) as f:
                self.data_dict = json.load(f)
        else:
            self.mode = "train"
            with open(
                os.path.join(
                    os.path.dirname(__file__), "../../resources/ucla_train.json"
                )
            ) as f:
                self.data_dict = json.load(f)

        self.nw_ucla_root = data_path
        self.time_steps = 52
        self.bone = [
            (1, 2),
            (2, 3),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 5),
            (7, 6),
            (8, 7),
            (9, 3),
            (10, 9),
            (11, 10),
            (12, 11),
            (13, 1),
            (14, 13),
            (15, 14),
            (16, 15),
            (17, 1),
            (18, 17),
            (19, 18),
            (20, 19),
        ]
        self.label = []
        for index in range(len(self.data_dict)):
            info = self.data_dict[index]
            self.label.append(int(info["label"]) - 1)

        self.debug = debug
        self.data_path = data_path
        self.label_path = mode
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        self.data = []
        for data in self.data_dict:
            file_name = data["file_name"]
            with open(self.nw_ucla_root + file_name + ".json", "r") as f:
                json_file = json.load(f)
            skeletons = json_file["skeletons"]
            value = np.array(skeletons)
            self.data.append(value)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = (
            data.mean(axis=2, keepdims=True)
            .mean(axis=4, keepdims=True)
            .mean(axis=0)
        )
        self.std_map = (
            data.transpose((0, 2, 4, 1, 3))
            .reshape((N * T * M, C * V))
            .std(axis=0)
            .reshape((C, 1, V, 1))
        )

    def __len__(self):
        return len(self.data_dict) * self.repeat

    def __iter__(self):
        return self

    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray(
            [
                [1, 0, 0],
                [0, math.cos(agx), math.sin(agx)],
                [0, -math.sin(agx), math.cos(agx)],
            ]
        )
        Ry = np.asarray(
            [
                [math.cos(agy), 0, -math.sin(agy)],
                [0, 1, 0],
                [math.sin(agy), 0, math.cos(agy)],
            ]
        )
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)]
        value = self.data[index % len(self.data_dict)]

        if self.mode == "train":
            random.random()
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)

            center = value[0, 1, :]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (
                np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0)
            )
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, 20, 3))

            data = np.zeros((self.time_steps, 20, 3))

            value = scalerValue[:, :, :]
            length = value.shape[0]

            random_idx = random.sample(
                list(np.arange(length)) * 100, self.time_steps
            )
            random_idx.sort()
            data[:, :, :] = value[random_idx, :, :]
            data[:, :, :] = value[random_idx, :, :]

        else:
            random.random()
            agx = 0
            agy = 0
            s = 1.0

            center = value[0, 1, :]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (
                np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0)
            )
            scalerValue = scalerValue * 2 - 1

            scalerValue = np.reshape(scalerValue, (-1, 20, 3))

            data = np.zeros((self.time_steps, 20, 3))

            value = scalerValue[:, :, :]
            length = value.shape[0]

            idx = np.linspace(0, length - 1, self.time_steps).astype(np.int32)
            data[:, :, :] = value[idx, :, :]  # T,V,C
        data = np.transpose(data, (2, 0, 1))
        C, T, V = data.shape
        data = np.reshape(data, (C, T, V, 1))

        data = data.astype(np.float32)
        sample = torch.from_numpy(data)
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

    def top_k(self, score, top_k):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

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
        return get_angular_motion(sample, 2, "ucla")
