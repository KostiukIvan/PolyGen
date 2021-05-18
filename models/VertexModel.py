import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def dequantize_vertices(vertices, *, n_bits=8, add_noise=False):
    """
    Convert quantized vertices to floats.
    """
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2 ** n_bits - 1
    vertices = vertices.astype('float32')
    vertices = vertices * (max_range - min_range) / range_quantize + min_range
    if add_noise:
        vertices += np.random.uniform(size=vertices.shape) * (1 / range_quantize)
    return vertices


# Those top functions aren't used with default values used in `VertexModel`
def top_k_logits(logits, k):
    """
    Masks logits such that logits not in top-k are small.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.Tensor(logits).to(dtype=torch.float32)

    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    k_largest = torch.min(values, dim=-2)[0].min()
    return torch.where(torch.le(logits, k_largest),
                       torch.ones_like(logits) * -1e9, logits)


# TODO does not work as tf version - probably needs to implemented from scratch [ look up top-p sampling]
def top_p_logits(logits, p):
    """
    Masks logits using nucleus (top-p) sampling.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.Tensor(logits).to(dtype=torch.float32)

    if p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -1e+9
        return logits


def sample_top_p(logits, top_p):
    """
    Samples random index of logits tensor using top-p sampling
    Args:
        logits (Tensor): logits distribution
        top_p (float): p value of sampling (0 <= p <= 1)
    Returns:
        Tensor: randomly picked index
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumsum = torch.cumsum(sorted_logits, dim=-1)
    tensors_amount_to_remove = len(sorted_indices[cumsum >= top_p])
    if tensors_amount_to_remove > 0:
        tensors_amount_to_remove -= 1
    indices_to_pick_from = sorted_indices if tensors_amount_to_remove == 0 else sorted_indices[:-tensors_amount_to_remove]
    return indices_to_pick_from[random.randint(0, len(indices_to_pick_from) - 1)]


class VertexModel(nn.Module):
    """
    Autoregressive generative model of quantized mesh vertices.
    Operates on flattened vertex sequences with a stopping token:
    [z_0, y_0, x_0, z_1, y_1, x_1, ..., z_n, y_n, z_n, STOP]

    Input vertex coordinates are embedded and tagged with learned coordinate and
    position indicators. A transformer decoder outputs logits for a quantized
    vertex distribution.
    """

    def __init__(self,
                 decoder,
                 embedding_dim,
                 quantization_bits,
                 class_conditional=False,
                 num_classes=55,
                 max_num_input_verts=1000,
                 use_discrete_embeddings=True,
                 device=None):
        """
        Parameters
        ----------
        decoder: dict
            TransformerDecoder object.
        embedding_dim: int
            Value taken from TransformerDecoder and it determines used embedding dim.
        quantization_bits: int
            Determines number of quantization used in mesh preprocessing.
        class_conditional: bool, optional
            If True, then condition on learned class embeddings.
        num_classes: int, optional
            Number of classes to condition on.
        max_num_input_verts: int, optional
            Maximum number of vertices, value used for learned positional embeddings.
        use_discrete_embeddings: bool, optional
            If True use discrete embeddings, otherwise use continuous.
        """
        super().__init__()
        self.decoder = decoder
        self.embedding_dim = embedding_dim
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_num_input_verts = max_num_input_verts
        self.quantization_bits = quantization_bits
        self.use_discrete_embeddings = use_discrete_embeddings
        self.device = device

        # prepare embeddings modules
        self.global_context_embedding = nn.Embedding(
            num_embeddings=self.num_classes,
            embedding_dim=self.embedding_dim
        ) if self.class_conditional else None

        self.coord_embeddings = nn.Embedding(4, self.embedding_dim, padding_idx=0)
        self.pos_embeddings = nn.Embedding(self.max_num_input_verts, self.embedding_dim, padding_idx=0)
        self.vert_embedding = nn.Embedding(self.embedding_dim + 3, self.embedding_dim, padding_idx=2)

        self.project_to_logits = nn.LayerNorm(self.embedding_dim)

    def _embed_inputs(self, batch_d, targets=None):
        """
        Embeds vertex positions, values and coordinate info

        Parameters
        ----------
        vertices: torch.Tensor
            Vertices of shape [batch_size, max_sequence_length, 3]
        targets: torch.Tensor
            TODO: make use of it in the future
            Tensor of labels required if `class_conditional` is True.
        """

        coord_embeddings = self.coord_embeddings(batch_d['axises_tokens'].long().to(self.device))
        pos_embeddings = self.pos_embeddings(batch_d['position_tokens'].long().to(self.device))
        vert_embeddings = self.vert_embedding(batch_d['vertices_tokens'].long().to(self.device))
        embeddings = vert_embeddings + coord_embeddings + pos_embeddings

        if self.global_context_embedding is None:
            batch_size = batch_d['vertices_tokens'].size(0)
            zero_embed = torch.zeros((1, 1, self.embedding_dim))
            zero_embed_tiled = torch.tile(zero_embed, (batch_size, 1, 1))
        else:
            zero_embed_tiled = self.global_context_embedding(targets).unsqueeze(1)


        return torch.cat([zero_embed_tiled, embeddings], dim=1)

    def forward(self, batch_d, *, targets=None,
                top_k=0, top_p=1):
        """
        Forward pass of VertexModel.

        Parameters
        ----------
        vertices: torch.Tensor
            Tensor representing batch of shape [batch_size, seq_len].
        Returns
        -------
        torch.Tensor
            Predictive distribution of shape [batch_size, seq_len]/
        targets: torch.Tensor, optional
            Tensor of labels required if `class_conditional` is True.
        top_k: int, optional
            Number of tokens to keep from top-k sampling.
        top_p: float, optional
            Proportion of probability mass to keep for top-p sampling.
        """

        embed = self._embed_inputs(batch_d, targets=targets)
        outputs = self.decoder(embed)

        outputs = self.project_to_logits(outputs)
        # logits /= temperature
        # logits = top_k_logits(logits, top_k)
        # logits = top_p_logits(logits, top_p)

        return outputs

    def sample(self, num_samples, *,
               context=None, max_sample_length=None, top_k=0, top_p=1, recenter_verts=True, only_return_complete=True):
        """
        Autoregressive sampling with caching.

        Parameters
        ----------
        num_samples: int
            Number of samples to produce.
        context:  torch.Tensor, optional
            Tensor of labels - provide class context to a model.
        max_sample_length: int
            Max len of sampled vertex sequences. Sequences that do not complete are truncated.
        top_k: int, optional
            Number of tokens to keep from top-k sampling.
        top_p: float, optional
            Proportion of probability mass to keep for top-p sampling.
        recenter_verts: bool, optional
            Center vertex samples around origin. (should be used if trained using shift augmentation)
        only_return_complete: bool, optional
            Determines if only completed samples should be returned.

        Returns
        -------
        dict
            Dictionary containing the following fields:
                - 'completed': Boolean tensor of shape [num_samples]. If True then
                   corresponding sample completed within max_sample_length.
                - 'vertices': Tensor of samples with shape [num_samples, num_verts, 3].
                - 'num_vertices': Tensor indicating number of vertices for each example
                                  in padded vertex samples.
                'vertices_mask': Tensor of shape [num_samples, num_verts] that masks
                                 corresponding invalid elements in 'vertices'.
        """
        if context is not None:
            global_context = self.global_context_embedding(context).detach()
            num_samples = torch.min(num_samples, global_context.size(0))

        # TODO implement