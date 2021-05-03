import torch.nn as nn
import torch


class VertexModel(nn.Module):
    """Autoregressive generative model of quantized mesh vertices.
      Operates on flattened vertex sequences with a stopping token:
      [z_0, y_0, x_0, z_1, y_1, x_1, ..., z_n, y_n, z_n, STOP]
      Input vertex coordinates are embedded and tagged with learned coordinate and
      position indicators. A transformer decoder outputs logits for a quantized
      vertex distribution.
      """
    def __init__(self, decoder, hidden_size, quantization_bits, class_conditional=False,  num_classes=55,
                 max_num_input_verts=2500, use_discrete_embeddings=True, name='vertex_model'):
        super().__init__()

        self.embedding_dim = hidden_size
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_num_input_verts = max_num_input_verts
        self.quantization_bits = quantization_bits
        self.use_discrete_embeddings = use_discrete_embeddings

        self.decoder = decoder
        self.coord_embeddings = nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim)
        self.pos_embeddings = nn.Embedding(num_embeddings=self.max_num_input_verts, embedding_dim=self.embedding_dim)
        self.vert_embeddings = nn.Embedding(num_embeddings=2**self.quantization_bits + 1, embedding_dim=self.embedding_dim)
        self.project_to_logits = nn.Linear(self.embedding_dim, 2**self.quantization_bits + 1) # + 1 for stopping token

    def forward(self, vertices: torch.Tensor()):
        # TODO : Add global context if required ??
        vertices = vertices.unsqueeze(0) if len(vertices.shape) < 3 else vertices
        batch_size = vertices.shape[0]
        vertices_flat = vertices.reshape(batch_size, -1)

        decoder_input = self._embed_inputs(vertices_flat)
        outputs = self.decoder(decoder_input)

        # TODO: compute logits and Categorical distribution
        logits = self.project_to_logits(outputs)


        return logits

    def _calcul_logits(self, decoder_outputs, temperature=1., top_k=0., top_p=1.):
        pass

    def _embed_inputs(self, vertices_flat: torch.Tensor()):
        """
        Embeds flat vertices and adds position and coordinate information.
        Args: vertices_flat: [batch_size, max_sequence_length]
        """
        # remove last element, as it is not used in embedding processs
        vertices_flat = vertices_flat[:, :-1]
        batch_size, seq_length = vertices_flat.shape[0], vertices_flat.shape[1]

        coord_embed_input = torch.range(seq_length) % 3
        coord_embed = self.coord_embeddings(coord_embed_input)

        pos_embed_input = torch.range(seq_length) // 3
        pos_embed = self.pos_embeddings(pos_embed_input)

        vert_embed = self.vert_embeddings(vertices_flat)

        # aggregate embeddings
        # TODO: Add zero embedding tiled
        embeddings = vert_embed + (coord_embed + pos_embed)
        return embeddings
