import numpy as np
import torch
import torch.nn.functional as F


def moving_augment(
    sample: np.ndarray,
    scale_range=(-0.5, 0.5),
    translate_range=(0, 0),
):
    scale = (
        np.random.rand() * (scale_range[1] - scale_range[0])
        + scale_range[0]
        + 1
    )
    translate = (
        np.random.rand() * (translate_range[1] - translate_range[0])
        + translate_range[0]
    )
    C, T, V, M = sample.shape
    scale_arr = np.linspace(1, scale, T)
    translate_arr = np.linspace(0, translate, T)
    scale_arr = np.tile(scale_arr, (C, 1))
    translate_arr = np.tile(translate_arr, (C, 1))

    aug_sample = (sample.T * scale_arr.T + translate_arr.T).T

    return aug_sample


class UniSampling(object):
    def __init__(self, new_length: int):
        self.new_length = new_length

    def __call__(self, sample: np.ndarray, train=True):
        C, T, V, M = sample.shape

        chunk_size = T / self.new_length
        aug_ids = np.linspace(0, T, self.new_length + 1)[:-1]
        if train:
            choosen = np.random.rand(self.new_length) * (chunk_size - 1)
        else:
            choosen = np.zeros(self.new_length)
        aug_ids = np.round(choosen + aug_ids).astype(np.int32)

        aug_sample = sample[:, aug_ids, :, :]

        return aug_sample


class ResizeSequence(object):
    def __init__(self, new_length: int, p_interval: list[float]):
        self.new_length = new_length
        self.p_interval = p_interval

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        C, T, V, M = data.size()
        begin = 0
        non_zero_frames = torch.nonzero(data.sum((0, 2, 3)))
        if non_zero_frames.numel() == 0:
            end = 1
        else:
            end = int(torch.max(non_zero_frames).item())
        valid_len = end - begin

        if len(self.p_interval) == 1:
            bias = int((1 - self.p_interval[0]) * valid_len / 2.0)
            data = data[:, begin + bias : end - bias, :, :]
            cropped_len = data.size(1)
        else:
            p = (
                np.random.rand() * (self.p_interval[1] - self.p_interval[0])
                + self.p_interval[0]
            )
            cropped_len = min(
                max(int(p * valid_len), self.new_length), valid_len
            )
            bias = np.random.randint(0, valid_len - cropped_len + 1)
            data = data[:, begin + bias : begin + bias + cropped_len, :, :]

        C, T, V, M = data.size()

        data = (
            data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_len)
        )
        data = data[None, None, :, :]
        data = F.interpolate(
            data,
            size=(C * V * M, self.new_length),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        data = (
            data.contiguous()
            .view(C, V, M, self.new_length)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return data


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros), dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim=1)  # T,3,3

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


class RandomRotate(object):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, data):
        theta = self.theta

        C, T, V, M = data.shape
        data = (
            data.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)
        )  # T,3,V*M
        rot = torch.zeros(3).uniform_(-theta, theta)
        rot = torch.stack(
            [
                rot,
            ]
            * T,
            dim=0,
        )
        rot = _rot(rot)  # T,3,3
        data = torch.matmul(rot, data)
        data = (
            data.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()
        )

        return data
