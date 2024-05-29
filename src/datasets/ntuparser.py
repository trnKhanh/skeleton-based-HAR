import os
import numpy as np

import torch
from torch.utils.data import Dataset
from src.datasets.augment import ResizeSequence
from src.datasets.utils import get_angular_motion

from src.graph.ntu_graph import Graph
from src.datasets.utils import get_angular_motion
from src.preprocess.raw_data import load_skeleton_data, convert_to_numpy
from src.preprocess.denoise_data import get_denoise_data


class NTUParser(object):

    def __init__(
        self,
        length_t=64,
        features="j",
        center=20,
    ):
        super().__init__()
        self.graph = Graph(center)

        self.transform = ResizeSequence(length_t)
        self.features = features

    def get_sample(self, sample_path) -> torch.Tensor:
        if os.path.splitext(sample_path)[1] == ".skeleton":
            data = load_skeleton_data(sample_path)
            bodies_data = convert_to_numpy(data)
            bodies_data["name"] = sample_path.split("/")[-1]

            sample = get_denoise_data(bodies_data)
        else:
            sample = np.load(sample_path, allow_pickle=True)

        sample = torch.from_numpy(sample)

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

        if data is not None and self.transform is not None:
            data = self.transform(data)
        if data is None:
            raise ValueError(f"Features {self.features} are invalid")
        return data

    def __get_motion(self, sample: torch.Tensor):
        C, T, V, M = sample.size()
        diff = sample[:, 1:, :, :] - sample[:, :-1, :, :]

        joints_motion = torch.zeros_like(sample)
        joints_motion[:, 1:, :, :] = diff

        return joints_motion

    def __get_bones(self, sample: torch.Tensor):
        C, T, V, M = sample.size()
        bones = torch.zeros_like(sample)

        A = self.graph.get_A()
        num_joints = A.shape[0]
        for u in range(num_joints):
            for v in range(num_joints):
                if A[u, v] == 0:
                    continue
                if self.graph.depth[u] > self.graph.depth[v]:
                    bones[:, :, u, :] = sample[:, :, u, :] - sample[:, :, v, :]
        return bones

    def __get_bones_motion(self, sample: torch.Tensor):
        C, T, V, M = sample.size()
        bones = self.__get_bones(sample)

        bones_motion = self.__get_motion(bones)

        return bones_motion

    def __get_angular_motion(self, sample: torch.Tensor):
        return get_angular_motion(sample)
