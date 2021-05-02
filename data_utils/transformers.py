import numpy as np


class SortVertices:
    """
        This class is responsible for sorting vertices as was provided in PolyGen paper.
    """
    def __init__(self):
        pass

    def __call__(self, vertices):
        """
        Args:
            vertices (tensor (n_vertices, (x, y, z): tensor of vertices
        """
        """Load obj file and process."""
        # sorting vertices
        vertices_keys = [vertices[..., i] for i in range(vertices.shape[-1])]
        sort_idxs = np.lexsort(vertices_keys)
        vertices = vertices[sort_idxs]
        return vertices


class NormalizeVertices:
    def __init__(self):
        pass

    def __call__(self, vertices):
        # normalize vertices to range [0.0, 1.0]
        limits = [-1.0, 1.0]
        vertices = (vertices - limits[0]) / (limits[1] - limits[0])
        return vertices


class QuantizeVertices:
    def __init__(self):
        pass

    def __call__(self, vertices):
        # quantize vertices to integers in range [0, 255]
        n_vals = 2 ** 8
        delta = 1. / n_vals
        vertices = np.maximum(np.minimum((vertices // delta), n_vals - 1), 0).astype(np.int32)
        return vertices



