import numpy as np
import torch



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
        # sorting vertices
        vertices_keys = [vertices[..., i] for i in range(vertices.shape[-1])]
        sort_idxs = np.lexsort(vertices_keys)
        vertices = vertices[sort_idxs]
        return vertices


class SortVerticesZOrder:
    """
        Sorts vertices using z-order curve
    """
    def __init__(self):
        pass
    
    @staticmethod
    def _less_msb(x: int, y: int) -> bool:
        return x < y and x < (x ^ y)

    @staticmethod
    def _cmp_zorder(lhs, rhs) -> bool:
        """Compare z-ordering."""
        # Assume lhs and rhs array-like objects of indices.
        assert len(lhs) == len(rhs)
        # Will contain the most significant dimension.
        msd = 0
        # Loop over the other dimensions.
        for dim in range(1, len(lhs)):
            # Check if the current dimension is more significant
            # by comparing the most significant bits.
            if SortVerticesZOrder._less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
                msd = dim
        return lhs[msd] < rhs[msd]

    @staticmethod
    def _partition(arr, low, high):
        i = (low-1)
        pivot = arr[high]
    
        for j in range(low, high):
            if not SortVerticesZOrder._cmp_zorder(arr[j], pivot):
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
    
        arr[i+1], arr[high] = arr[high], arr[i+1]
        return (i+1)
    
    @staticmethod
    def _quickSort(arr, low, high):
        if len(arr) == 1:
            return arr
        if low < high:
            pi = SortVerticesZOrder._partition(arr, low, high)
            SortVerticesZOrder._quickSort(arr, low, pi-1)
            SortVerticesZOrder._quickSort(arr, pi+1, high)

    def __call__(self, vertices):
        """
        Args:
            vertices [n_vertices, [x, y, z]]: array of vertices
        """
        vertices = list(vertices)
        SortVerticesZOrder._quickSort(vertices, 0, len(vertices) - 1)
        return np.array(vertices)


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



def detokenize(vertices_tokens):
    return torch.reshape(vertices_tokens, shape=(-1, 3))


def extract_vert_values_from_tokens(vert_tokens, seq_len=2400, is_target=False):
    # note: Targets needs to be extracted in order to calculate chamfer dist.
    if not is_target:
        vert_tokens = torch.max(vert_tokens, dim=0)[1]

    vertices = detokenize(vert_tokens) #[: ((seq_len - 2) // 3) * 3])
    vertices = vertices.float()
    vertices /= 256
    return vertices
