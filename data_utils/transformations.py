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
    def __init__(self, max_seq_len=1200):
        self.max_seq_len = max_seq_len - 2 # make one slot left for eos token
        self.tokens = {
            'bos': torch.tensor([0]),
            'pad': torch.tensor([1]),
            'eos': torch.tensor([2]),
        }
    
    def __call__(self, vertices):
        vertices_tokens = torch.flatten(vertices)
        axises_tokens = torch.arange(len(vertices_tokens)) % 3
        position_tokens = torch.arange(len(vertices_tokens)) // 3
        
        if len(vertices_tokens) < self.max_seq_len:
            amount_to_pad = self.max_seq_len - len(vertices_tokens)
            vertices_tokens = pad(vertices_tokens, (0, amount_to_pad), value=self.tokens['pad'][0])
            axises_tokens = pad(axises_tokens, (0, amount_to_pad), value=self.tokens['pad'][0])
            position_tokens = pad(position_tokens, (0, amount_to_pad), value=self.tokens['pad'][0])

        vertices_tokens = torch.cat([self.tokens['bos'], vertices_tokens, self.tokens['eos']])
        axises_tokens = torch.cat([self.tokens['bos'], axises_tokens, self.tokens['eos']])
        position_tokens = torch.cat([self.tokens['bos'], position_tokens, self.tokens['eos']])

        return {"vertices_tokens": vertices_tokens,
                "axises_tokens": axises_tokens,
                "position_tokens": position_tokens}


def detokenize(vertices_tokens):
    return torch.reshape(vertices_tokens, shape=(-1, 3))

def extract_vert_values_from_tokens(vert_tokens, seq_len=2400):
    vert_tokens = torch.max(vert_tokens[1:(seq_len - 1),:], dim=1)[1]
    vertices = detokenize(vert_tokens[: ((seq_len - 2) // 3) * 3])
    vertices = vertices.float()
    vertices /= 256
    return vertices