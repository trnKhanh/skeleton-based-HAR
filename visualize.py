from argparse import ArgumentParser
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
from torch import nn

from src.models.net import STGCN
from src.datasets.ntuparser import NTUParser
from src.graph.ntu_graph import Graph


def create_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--device", default="cpu", type=str, help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--sample-path", required=True, type=str, help="Path to sample file"
    )
    parser.add_argument(
        "--load-ckpt", default="", type=str, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--features",
        default=["j"],
        nargs="+",
        type=str,
        help="Features to use as inputs. If model ensemble is used, this should match the order of given models",
    )
    parser.add_argument(
        "--ensemble",
        default=[],
        nargs="*",
        type=str,
        help="List of models used in multi-stream",
    )
    parser.add_argument(
        "--alphas",
        default=[],
        nargs="*",
        type=float,
        help="weight for each model when ensembling",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Whether to use adaptive graph for graph edges",
    )
    parser.add_argument(
        "--num-classes",
        default=60,
        type=int,
        help="Number of classes (default: 60)",
    )
    parser.add_argument(
        "--length-t",
        default=64,
        type=int,
        help="Number of frames in each sample",
    )

    return parser.parse_args()


def main(args):
    classes_path = "./resources/ntu_classes.txt"
    class_map = dict()
    with open(classes_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            id, name = line.split(".")[:2]
            id = int(id) - 1
            name = name.strip()
            class_map[id] = name

    args.device = torch.device(args.device)
    models = nn.ModuleList()
    if len(args.load_ckpt) > 0:
        args.ensemble = [args.load_ckpt]

    for i in range(len(args.ensemble)):
        num_features = args.features[i].count(",") + 1
        models.append(
            STGCN(
                3 * num_features,
                args.num_classes,
                act_layer=nn.ReLU,
                adaptive=args.adaptive,
            )
        )

        state_dict = torch.load(args.ensemble[i], map_location=args.device)
        if "model" in state_dict:
            models[-1].load_state_dict(state_dict["model"])
        else:
            models[-1].load_state_dict(state_dict)
        print(f"Loaded model from {args.ensemble[i]}")
    models.to(args.device)
    models.eval()

    with torch.no_grad():
        scores = torch.zeros(0)
        for i in range(len(models)):
            parser = NTUParser(
                features=args.features[i], length_t=args.length_t
            )

            sample = parser.get_sample(args.sample_path)
            sample = sample.unsqueeze(0)
            sample = sample.to(args.device)

            preds = models[i](sample)
            if len(args.alphas) == 0:
                scores = preds if scores.size(0) == 0 else scores + preds
            else:
                scores = (
                    args.alphas[i] * preds
                    if scores.size(0) == 0
                    else scores + args.alphas[i] * preds
                )

        pred_class = torch.argmax(scores).item()

    ntu_graph = Graph()

    def update(num, ax, data):
        ax.cla()
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 7)
        ax.set_zlim(-1, 1)
        A = ntu_graph.get_A()

        for m in range(2):
            for u in range(A.shape[0]):
                for v in range(A.shape[1]):
                    if A[u][v] == 0:
                        continue
                    x = [data[0, num, u, m], data[0, num, v, m]]
                    y = [data[1, num, u, m], data[1, num, v, m]]
                    z = [data[2, num, u, m], data[2, num, v, m]]
                    if x[0] == x[1] and y[0] == y[1] and z[0] == z[1]:
                        continue
                    ax.plot(x, z, y, color="b")

    parser = NTUParser(length_t=args.length_t, features="j")
    sample = parser.get_sample(args.sample_path)
    fig = plt.figure()
    fig.canvas.manager.set_window_title(class_map[pred_class])
    ax = fig.add_subplot(111, projection="3d")
    ani = animation.FuncAnimation(
        fig, func=update, frames=args.length_t, fargs=[ax, sample], interval=75
    )
    plt.show()
    print(f"Prediction: {class_map[pred_class]}")


if __name__ == "__main__":
    args = create_args()
    main(args)
