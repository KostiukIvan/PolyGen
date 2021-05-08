import numpy as np
import torch
from torch.nn.functional import pad

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


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, vertices):
        return torch.from_numpy(vertices)


class ResizeVertices:
    def __init__(self, target_len_of_seq=600):
        self.target_len_of_seq = target_len_of_seq

    def __call__(self, vertices):
        curr_len_of_seq = vertices.shape[0]
        if curr_len_of_seq >= self.target_len_of_seq:
            vertices = vertices[:self.target_len_of_seq, :]
        else:
            diff_len_of_seq = self.target_len_of_seq - curr_len_of_seq
            zeros = torch.zeros((diff_len_of_seq, 3))
            vertices = torch.cat([vertices, zeros], dim=0)

        return vertices


class VertexTokenizer:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len - 1 # make one slot left for eos token
    
    def __call__(self, vertices):
        vertices_tokens = torch.flatten(vertices) + 1
        axises_tokens = torch.arange(len(vertices_tokens)) % 3 + 1
        position_tokens = torch.arange(len(vertices_tokens)) // 3 + 1        
        
        if len(vertices) < self.max_seq_len:
            amount_to_pad = self.max_seq_len * 3 - len(vertices) * 3
            vertices_tokens = pad(vertices_tokens, (0, amount_to_pad), value=0)
            axises_tokens = pad(axises_tokens, (0, amount_to_pad), value=0)
            position_tokens = pad(position_tokens, (0, amount_to_pad), value=0)
        
        return torch.stack([vertices_tokens, axises_tokens, position_tokens])



