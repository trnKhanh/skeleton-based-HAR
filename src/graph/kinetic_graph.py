import sys
import numpy as np

sys.path.extend(["../"])
from src.graph import tools

num_node = 18
self_link = [(i, i) for i in range(num_node)]
inward = [
    (4, 3),
    (3, 2),
    (7, 6),
    (6, 5),
    (13, 12),
    (12, 11),
    (10, 9),
    (9, 8),
    (11, 5),
    (8, 2),
    (5, 1),
    (2, 1),
    (0, 1),
    (15, 0),
    (14, 0),
    (17, 15),
    (16, 14),
]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode="spatial"):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
