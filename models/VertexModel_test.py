import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from reformer_pytorch import Reformer


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
    if type(m) == nn.Embedding:
        nn.init.uniform_(m.weight, -0.05, 0.05)


def accuracy(y_pred, y_true, ignore_label=None, device=None):
    y_pred = y_pred.argmax(dim=1)

    if ignore_label:
        normalizer = torch.sum(y_true!=ignore_label)
        ignore_mask = torch.where(
            y_true == ignore_label,
            torch.zeros_like(y_true, device=device),
            torch.ones_like(y_true, device=device)
        ).type(torch.float32)
    else:
        normalizer = y_true.shape[0]
        ignore_mask = torch.ones_like(y_true, device=device).type(torch.float32)

    acc = (y_pred.reshape(-1)==y_true.reshape(-1)).type(torch.float32)
    acc = torch.sum(acc*ignore_mask)
    return acc / normalizer

class VertexDecoderEmbedding(nn.Module):

    def __init__(self, embed_dim=256,
                 vocab_value=259, pad_idx_value=2,
                 vocab_coord_type=4, pad_idx_coord_type=0,
                 vocab_position=1000, pad_idx_position=0, device=None):
        super().__init__()

        self.value_embed = nn.Embedding(
            vocab_value, embed_dim, padding_idx=pad_idx_value
        )
        self.coord_type_embed = nn.Embedding(
            vocab_coord_type, embed_dim, padding_idx=pad_idx_coord_type
        )
        self.position_embed = nn.Embedding(
            vocab_position, embed_dim, padding_idx=pad_idx_position
        )

        self.embed_scaler = math.sqrt(embed_dim)
        self.device = device

    def forward(self, tokens):
        """get embedding for vertex model.

        Args
            tokens [dict]: tokenized vertex info.
                `value_tokens` [torch.tensor]:
                        padded (batch, length)-shape long tensor
                        with coord value from 0 to 2^n(bit).
                `coord_type_tokens` [torch.tensor]:
                        padded (batch, length) shape long tensor implies x or y or z.
                `position_tokens` [torch.tensor]:
                        padded (batch, length) shape long tensor
                        representing coord position (NOT sequence position).

        Returns
            embed [torch.tensor]: (batch, length, embed) shape tensor after embedding.

        """
        embed = self.value_embed(tokens["value_tokens"].to(self.device).long()) * self.embed_scaler
        embed = embed + (self.coord_type_embed(tokens["coord_type_tokens"].to(self.device)) * self.embed_scaler)
        embed = embed + (self.position_embed(tokens["position_tokens"].to(self.device)) * self.embed_scaler)

        return embed

class VertexPolyGen(nn.Module):
    """Vertex model in PolyGen.
    this model learn/predict vertices like OpenAI-GPT.
    UNLIKE the paper, this model is only for unconditional generation.

    Args
        model_config [Config]:
                hyper parameters. see VertexPolyGenConfig class for details.
    """

    def __init__(self, model_config, device):
        super().__init__()

        self.tokenizer = DecodeVertexTokenizer(**model_config["tokenizer"])
        self.embedding = VertexDecoderEmbedding(**model_config["embedding"], device=device)
        self.reformer = Reformer(**model_config["reformer"])
        self.layernorm = nn.LayerNorm(model_config["embed_dim"])
        self.loss_func = nn.CrossEntropyLoss(ignore_index=model_config["tokenizer"]["pad_id"])

        self.apply(init_weights)


    def forward(self, tokens, device=None):

        """forward function which can be used for both train/predict.

        Args
            tokens [dict]: tokenized vertex info.
                `value_tokens` [torch.tensor]:
                        padded (batch, length)-shape long tensor
                        with coord value from 0 to 2^n(bit).
                `coord_type_tokens` [torch.tensor]:
                        padded (batch, length) shape long tensor implies x or y or z.
                `position_tokens` [torch.tensor]:
                        padded (batch, length) shape long tensor
                        representing coord position (NOT sequence position).
                `padding_mask` [torch.tensor]:
                        (batch, length) shape mask implies <pad> tokens.
            device [torch.device]: gpu or not gpu, that's the problem.


        Returns
            hs [torch.tensor]:
                    hidden states from transformer(reformer) model.
                    this takes (batch, length, embed) shape.

        """

        hs = self.embedding(tokens)
        
        hs = self.reformer(
            hs, input_mask=tokens["padding_mask"].to(device)
        )
        hs = self.layernorm(hs)

        return hs

    def __call__(self, inputs, device=None):

        """Calculate loss while training.

        Args
            inputs [dict]: dict containing batched inputs.
                `vertices` [list(torch.tensor)]:
                        variable-length-list of
                        (length, 3) shaped tensor of quantized-vertices.
            device [torch.device]: gpu or not gpu, that's the problem.

        Returns
            outputs [dict]: dict containing calculated variables.
                `loss` [torch.tensor]:
                        calculated scalar-shape loss with backprop info.
                `accuracy` [torch.tensor]:
                        calculated scalar-shape accuracy.

        """

        tokens = self.tokenizer.tokenize(inputs)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        hs = self.forward(tokens, device=device)

        hs = F.linear(hs, self.embedding.value_embed.weight)
        BATCH, LENGTH, EMBED = hs.shape
        hs = hs.reshape(BATCH * LENGTH, EMBED)
        targets = tokens["target_tokens"].reshape(BATCH * LENGTH, )

        acc = accuracy(
            hs, targets, ignore_label=self.tokenizer.pad_id, device=device
        )
        
        loss = self.loss_func(hs, targets.to(device).long())

        if hasattr(self, 'reporter'):
            self.reporter.report({
                "accuracy": acc.item(),
                "perplexity": torch.exp(loss).item(),
                "loss": loss.item(),
            })

        return loss, acc, hs

    @torch.no_grad()
    def predict(self, max_seq_len=2400, device=None):
        """predict function

        Args
            max_seq_len[int]: max sequence length to predict.
            device [torch.device]: gpu or not gpu, that's the problem.

        Return
            preds [torch.tensor]: predicted (length, ) shape tensor.

        """

        tokenizer = self.tokenizer
        special_tokens = tokenizer.special_tokens

        tokens = tokenizer.get_pred_start()
        tokens = {k: v.to(device) for k, v in tokens.items()}
        preds = []
        pred_idx = 0

        while (pred_idx <= max_seq_len - 1) \
                and ((len(preds) == 0) or (preds[-1] != special_tokens["eos"] - len(special_tokens))):

            if pred_idx >= 1:
                tokens = tokenizer.tokenize([torch.stack(preds)])
                tokens["value_tokens"][:, pred_idx + 1] = special_tokens["pad"]
                tokens["padding_mask"][:, pred_idx + 1] = True

            hs = self.forward(tokens, device=device)

            hs = F.linear(hs[:, pred_idx], self.embedding.value_embed.weight)
            pred = hs.argmax(dim=1) - len(special_tokens)
            preds.append(pred[0])
            pred_idx += 1

        preds = torch.stack(preds) + len(special_tokens)
        preds = self.tokenizer.detokenize([preds])[0]
        return preds


class Tokenizer(object):

    def _padding(self, ids_tensor, pad_token, max_length=None):
        if max_length is None:
            max_length = max([len(ids) for ids in ids_tensor])

        ids_tensor = [
            torch.cat([
                ids, pad_token.repeat(max_length - len(ids) + 1)
            ])
            for ids in ids_tensor
        ]
        return ids_tensor

    def _make_padding_mask(self, ids_tensor, pad_id):
        mask = torch.where(
            ids_tensor == pad_id,
            torch.ones_like(ids_tensor),
            torch.zeros_like(ids_tensor)
        ).type(torch.bool)
        return mask

    def _make_future_mask(self, ids_tensor):
        batch, length = ids_tensor.shape
        arange = torch.arange(length)
        mask = torch.where(
            arange[None, :] <= arange[:, None],
            torch.zeros((length, length)),
            torch.ones((length, length)) * (-np.inf)
        ).type(torch.float32)
        return mask

    def get_pred_start(self, start_token="bos", batch_size=1):
        special_tokens = self.special_tokens
        not_coord_token = self.not_coord_token
        max_seq_len = self.max_seq_len

        values = torch.stack(
            self._padding(
                [special_tokens[start_token]] * batch_size,
                special_tokens["pad"],
                max_seq_len
            )
        )
        coord_type_tokens = torch.stack(
            self._padding(
                [self.not_coord_token] * batch_size,
                not_coord_token,
                max_seq_len
            )
        )
        position_tokens = torch.stack(
            self._padding(
                [self.not_coord_token] * batch_size,
                not_coord_token,
                max_seq_len
            )
        )

        padding_mask = self._make_padding_mask(values, self.pad_id)

        outputs = {
            "value_tokens": values,
            "coord_type_tokens": coord_type_tokens,
            "position_tokens": position_tokens,
            "padding_mask": padding_mask,
        }
        return outputs


class DecodeVertexTokenizer(Tokenizer):

    def __init__(self, bos_id=0, eos_id=1, pad_id=2, max_seq_len=None):

        self.special_tokens = {
            "bos": torch.tensor([bos_id]),
            "eos": torch.tensor([eos_id]),
            "pad": torch.tensor([pad_id]),
        }
        self.pad_id = pad_id
        self.not_coord_token = torch.tensor([0])
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len - 1
        else:
            self.max_seq_len = max_seq_len

    def tokenize(self, vertices, padding=True):
        special_tokens = self.special_tokens
        not_coord_token = self.not_coord_token
        max_seq_len = self.max_seq_len

        vertices = [
            torch.cat([
                special_tokens["bos"],
                v.reshape(-1, ) + len(special_tokens),
                special_tokens["eos"]
            ])
            for v in vertices
        ]

        coord_type_tokens = [
            torch.cat([
                not_coord_token,
                torch.arange(len(v) - 2) % 3 + 1,
                not_coord_token
            ])
            for v in vertices
        ]

        position_tokens = [
            torch.cat([
                not_coord_token,
                torch.arange(len(v) - 2) // 3 + 1,
                not_coord_token
            ])
            for v in vertices
        ]

        vertices_target = [
            torch.cat([v, special_tokens["pad"]])[1:]
            for v in vertices
        ]

        if padding:
            vertices = torch.stack(
                self._padding(vertices, special_tokens["pad"], max_seq_len)
            )
            vertices_target = torch.stack(
                self._padding(vertices_target, special_tokens["pad"], max_seq_len)
            )
            coord_type_tokens = torch.stack(
                self._padding(coord_type_tokens, not_coord_token, max_seq_len)
            )
            position_tokens = torch.stack(
                self._padding(position_tokens, not_coord_token, max_seq_len)
            )

            padding_mask = self._make_padding_mask(vertices, self.pad_id)
            # future_mask = self._make_future_mask(vertices)
            outputs = {
                "value_tokens": vertices,
                "target_tokens": vertices_target,
                "coord_type_tokens": coord_type_tokens,
                "position_tokens": position_tokens,
                "padding_mask": padding_mask,
                # "future_mask": future_mask,
            }
        else:
            outputs = {
                "value_tokens": vertices,
                "target_tokens": vertices_target,
                "coord_type_tokens": coord_type_tokens,
                "position_tokens": position_tokens,
            }

        return outputs

    def detokenize(self, vertices):
        special_tokens = self.special_tokens

        result = []
        for vertex in vertices:
            vertex = vertex - len(special_tokens)
            result.append(
                vertex[torch.where(vertex >= 0)]
            )
        return result


class EncodeVertexTokenizer(Tokenizer):

    def __init__(self, pad_id=0, max_seq_len=None):
        self.pad_token = torch.tensor([pad_id])
        self.pad_id = pad_id

        if max_seq_len is not None:
            self.max_seq_len = max_seq_len - 1
        else:
            self.max_seq_len = max_seq_len

    def tokenize(self, vertices, padding=True):
        max_seq_len = self.max_seq_len
        vertices = [v.reshape(-1, ) + 1 for v in vertices]
        coord_type_tokens = [torch.arange(len(v)) % 3 + 1 for v in vertices]
        position_tokens = [torch.arange(len(v)) // 3 + 1 for v in vertices]

        if padding:
            vertices = torch.stack(self._padding(vertices, self.pad_token, max_seq_len))
            coord_type_tokens = torch.stack(self._padding(coord_type_tokens, self.pad_token, max_seq_len))
            position_tokens = torch.stack(self._padding(position_tokens, self.pad_token, max_seq_len))
            padding_mask = self._make_padding_mask(vertices, self.pad_id)

            outputs = {
                "value_tokens": vertices,
                "coord_type_tokens": coord_type_tokens,
                "position_tokens": position_tokens,
                "padding_mask": padding_mask,
            }
        else:
            outputs = {
                "value_tokens": vertices,
                "coord_type_tokens": coord_type_tokens,
                "position_tokens": position_tokens,
            }

        return outputs
