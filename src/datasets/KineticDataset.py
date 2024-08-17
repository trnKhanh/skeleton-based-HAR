import os
import json
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from src.datasets.augment import ResizeSequence, RandomRotate
from src.datasets.utils import get_angular_motion

from src.graph.ntu_graph import Graph
from src.datasets.utils import get_angular_motion

from tqdm import tqdm


class KineticDataset(Dataset):
    def __init__(
        self,
        data_path,
        extra_data_path="",
        mode="train",
        length_t=64,
        features="j",
        center=1,
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
        if self.mode not in ["train", "valid"]:
            raise NameError(f"Mode {self.mode} is invalid")
        if self.mode == "valid":
            self.mode = "val"
        self.train_ids = []

        self.__read_data(data_path)
        if len(extra_data_path) > 0:
            self.__read_data(extra_data_path)

    def __read_data(self, path: str):
        print("-" * os.get_terminal_size().columns)
        print(f"Read {self.mode} data from {path}")
        with open(
            os.path.join(path, f"kinetics_{self.mode}_label.json"), "r"
        ) as f:
            label_info = json.load(f)
        for file in tqdm(
            sorted(
                os.scandir(os.path.join(path, f"kinetics_{self.mode}")),
                key=lambda x: x.name,
            ),
            desc=f"Process samples",
            ncols=0,
        ):
            sample_id = str(file.name).split(".")[0]
            has_ske = label_info[sample_id]["has_skeleton"]
            if not has_ske:
                continue
            label = label_info[sample_id]["label_index"]

            self.samples.append(file.path)
            self.labels.append(label)
        print("-" * os.get_terminal_size().columns)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label = self.labels[index]

        sample_path = self.samples[index]
        with open(sample_path, "r") as f:
            video_info = json.load(f)
        num_person_in = 5
        num_person_out = 2
        C = 3
        T = len(video_info["data"])
        V = 18
        print("$$$")

        # fill data_numpy
        data_numpy = np.zeros((C, T, V, num_person_in))
        for frame_info in video_info["data"]:
            frame_index = frame_info["frame_index"] - 1
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= num_person_in:
                    break
                pose = skeleton_info["pose"]
                score = skeleton_info["score"]
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[1:2] = -data_numpy[1:2]
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        label = video_info["label_index"]
        assert self.labels[index] == label

        # sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        print(data_numpy.shape)
        for t, s in enumerate(sort_index):
            print(data_numpy[:, t, :, :].shape)
            print(s)
            print(data_numpy[:, t, :, s].shape)
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
        print(data_numpy.shape)
        print("$$$")
        data_numpy = data_numpy[:, :, :, 0:num_person_out]
        data_numpy[2, :, :, :] = 0

        sample = torch.from_numpy(data_numpy)

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
        return get_angular_motion(sample, 1)
