import numpy as np
import torch
import torch.nn as nn


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
    def scatter_nd(indices, updates, shape):
        """
        PyTorch Implementation of the `tf.scatter_nd` function.
        """
        ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
        ndim = indices.size(-1)
        output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
        print(indices.shape, ndim)
        flatted_indices = torch.reshape(indices, (-1, ndim))
        slices = [flatted_indices[:, i] for i in range(ndim)]
        slices += [Ellipsis]
        ret[slices] = updates.view(*output_shape)
        return ret

    if not isinstance(logits, torch.Tensor):
        logits = torch.Tensor(logits).to(dtype=torch.float32)

    if p == 0:
        return logits
    _, seq, dim = logits.shape
    logits = torch.reshape(logits, (-1, dim))
    sort_indices = torch.argsort(logits, dim=-1, descending=True)
    probs = torch.softmax(logits, dim=0).gather(dim=1, index=sort_indices)
    cumprobs = torch.cumsum(probs, dim=-1)
    sort_mask = torch.greater(cumprobs, p).to(dtype=torch.int64)
    batch_indices = torch.tile(
        torch.range(0, logits.size(0) - 1).unsqueeze(-1), dims=[1, dim]
    ).to(dtype=torch.int64)
    top_p_mask = scatter_nd(
        indices=torch.stack([batch_indices, sort_indices], dim=-1),
        updates=sort_mask,
        shape=list(logits.shape)
    )
    print("torch", top_p_mask)
    logits -= top_p_mask * 1e9
    return torch.reshape(logits, (-1, seq, dim))


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
                 hidden_size,
                 quantization_bits, *,
                 class_conditional=False,
                 num_classes=55,
                 max_num_input_verts=2500,
                 use_discrete_embeddings=True):
        """
        Parameters
        ----------
        decoder: dict
            TransformerDecoder object.
        hidden_size: int
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
        self.embedding_dim = hidden_size
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_num_input_verts = max_num_input_verts
        self.quantization_bits = quantization_bits
        self.use_discrete_embeddings = use_discrete_embeddings

        # prepare embeddings modules
        self.global_context_embedding = nn.Embedding(
            num_embeddings=self.num_classes,
            embedding_dim=self.embedding_dim
        ) if self.class_conditional else None
        self.coord_embeddings = nn.Embedding(3, self.embedding_dim)
        self.pos_embeddings = nn.Embedding(self.max_num_input_verts, self.embedding_dim)
        if use_discrete_embeddings:
            self.vert_embedding = nn.Embedding(2 ** self.quantization_bits + 1, self.embedding_dim)
        else:
            self.vert_embedding = nn.Linear(max_num_input_verts, self.embedding_dim)
        self.project_to_logits = nn.Linear(self.embedding_dim, 2 ** self.quantization_bits + 1)

    def _embed_inputs(self, vertices, targets=None):
        """
        Embeds flat vertices and adds position and coordinate information.

        Parameters
        ----------
        vertices: torch.Tensor
            Flat vertices of shape [batch_size, max_sequence_length]
        targets: torch.Tensor
            Tensor of labels required if `class_conditional` is True.
        """
        # note: remove last element as it is not used for predictions
        vertices = vertices[:, :-1]
        batch_size, seq_length = vertices.size()

        if self.global_context_embedding is None:
            # TODO - check if works as `tf.get_variable`
            zero_embed = torch.zeros((1, 1, self.embedding_dim))
            zero_embed_tiled = torch.tile(zero_embed, (batch_size, 1, 1))
        else:
            zero_embed_tiled = self.global_context_embedding(targets)[:, None]

        embed_input = torch.range(seq_length)
        coord_embeddings = self.coord_embeddings(embed_input % 3)
        pos_embeddings = self.pos_embeddings(embed_input // 3)
        vert_embeddings = self.vert_embedding(vertices)
        embeddings = vert_embeddings + coord_embeddings + pos_embeddings

        return torch.cat([zero_embed_tiled, embeddings], dim=1)

    def forward(self, vertices, *, targets=None,
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
        outputs = self.decoder(self._embed_inputs(vertices, targets))

        logits = self.project_to_logits(outputs)
        # logits /= temperature
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)

        return logits

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
