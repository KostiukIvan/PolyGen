import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.tokenizer_vm import VertexTokenizer


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
    if k <= 0:
        return logits

    values, _ = torch.topk(logits, k)
    k_largest = torch.min(values, dim=-2)[0].min()
    return torch.where(torch.le(logits, k_largest),
                       torch.ones_like(logits) * -1e9, logits)


def top_p_logits(logits, top_p, min_tokens_to_keep=1, filter_value=-float('Inf')):
    """
    Samples random index of logits tensor using top-p sampling
    Args:
        logits (Tensor): logits distribution
        top_p (float): p value of sampling (0 <= p <= 1)
    Returns:
        Tensor: randomly picked index
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0

    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores = logits.masked_fill(indices_to_remove, filter_value)
    return scores


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
                 tokenizer,
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
        tokenizer: object with __call__ method
            VertexTokenizer object
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
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_num_input_verts = max_num_input_verts
        self.quantization_bits = quantization_bits
        self.use_discrete_embeddings = use_discrete_embeddings
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.tokens['pad'][0].item())
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

    def _embed_inputs(self, batch_d):
        """
        Embeds vertex positions, values and coordinate info

        Parameters
        ----------
        vertices: torch.Tensor
            Vertices of shape [batch_size, max_sequence_length, 3]
ue.
        """
        
        coord_embeddings = self.coord_embeddings(batch_d['axises_tokens'].long().to(self.device))
        pos_embeddings = self.pos_embeddings(batch_d['position_tokens'].long().to(self.device))
        vert_embeddings = self.vert_embedding(batch_d['vertices_tokens'].long().to(self.device))
        embeddings = vert_embeddings + coord_embeddings + pos_embeddings

        return embeddings
        if self.global_context_embedding is None:
            batch_size = batch_d['vertices_tokens'].size(0)
            zero_embed = torch.zeros((1, 1, self.embedding_dim))
            zero_embed_tiled = torch.tile(zero_embed, (batch_size, 1, 1))
        else:
            zero_embed_tiled = self.global_context_embedding(targets.to(self.device)).unsqueeze(1)

        return torch.cat([zero_embed_tiled, embeddings], dim=1)

    def forward(self, batch_d, top_k=0, top_p=1):
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
        top_k: int, optional
            Number of tokens to keep from top-k sampling.
        top_p: float, optional
            Proportion of probability mass to keep for top-p sampling.
        """
        
        embed = self._embed_inputs(batch_d)
        outputs = self.decoder(embed, input_mask=batch_d['padding_mask'])

        outputs = self.project_to_logits(outputs)

        # outputs /= temperature
        outputs = top_k_logits(outputs, top_k)
        outputs = top_p_logits(outputs, top_p)

        outputs = torch.transpose(outputs, 1, 2)
        return outputs

    def backward(self, model_output, target_vertices):
        """
            Backward pass of VertexModel.

            Parameters
            ----------
            model_output: torch.Tensor
                Tensor representing batch of shape [batch_size, seq_len].
                Output returned by VertexModel
            target_vertices: torch.Tensor
                Tensor no grad representing batch of shape [batch_size, seq_len].
                Target output of the current object
            Returns
            -------
            torch.Tensor
                Value of the calculated loss
        """
        assert model_output.shape[0] == target_vertices.shape[0]

        loss = self.loss(model_output, target_vertices)
        return loss

    def sample(self, num_samples, tokenizer, *,
               context=None, max_sample_length=None, top_k=0, top_p=1):
        """
        Autoregressive sampling.

        Parameters
        ----------
        num_samples: int
            Number of samples to produce.
        tokenizer: VertexTokenizer
            Tokenizer required for the generation of initial tokens.
        context:  torch.Tensor, optional
            Tensor of labels - provide class context to a model.
        max_sample_length: int
            Max len of sampled vertex sequences. Sequences that do not complete are truncated.
        top_k: int, optional
            Number of tokens to keep from top-k sampling.
        top_p: float, optional
            Proportion of probability mass to keep for top-p sampling.

        Returns
        -------
        torch.Tensor
            Generated tensor of vertices.
        """
        max_sample_length = max_sample_length or self.max_num_input_verts
        init_len = 1
        tokens_d, pred_idx = self.tokenizer.get_initial_sampling_tokens(num_samples, init_len)
        tokens_d = {k: v.to(self.device) for k, v in tokens_d.items()}

        preds = []
        while pred_idx <= max_sample_length - 1:
            if pred_idx >= (1 + init_len):
                #print(pred.shape, pred)
                one_prediction = pred[:, pred_idx - 1]
                #print("0: ",  tokens_d['vertices_tokens'].shape, tokens_d['vertices_tokens'])
                pred = tokens_d['vertices_tokens'][:, 1 : pred_idx + 1]
                #print("1: ", pred.shape, pred)
                pred[:, pred_idx - 1] = one_prediction
                print("2: ", pred.shape, pred)

                tokens_d = self.tokenizer.tokenize_without_end(pred)
                #print("3: ", tokens_d['vertices_tokens'].shape, tokens_d['vertices_tokens'])
                #if pred_idx > 3:
                #    return
                tokens_d = {k: v.unsqueeze(0) for k, v in tokens_d.items()}
                #tokens_d['vertices_tokens'][:, pred_idx + 1:] = self.tokenizer.tokens['pad']
                #tokens_d['padding_mask'][:, pred_idx + 1:] = True


            recon_tokenized_vertices = self.forward(tokens_d)
            pred = torch.max(recon_tokenized_vertices, dim=1)[1]
            
            if pred[:, pred_idx] == self.tokenizer.tokens['eos'] or pred_idx >= max_sample_length:
                break

            pred_idx += 1

        #preds = torch.stack(pred)# + len(tokenizer.tokens)
        return pred
