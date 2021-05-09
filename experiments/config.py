from pathlib import Path
from data_utils.transformations import *


class Config(object):
    def __getitem__(self, key):
        return self.config[key]


class VertexConfig(Config):

    def __init__(self,
                 embed_dim=256,
                 max_seq_len=2400,
                 tokenizer__bos_id=0,
                 tokenizer__eos_id=1,
                 tokenizer__pad_id=2,
                 embedding__vocab_value=256 + 3,
                 embedding__vocab_coord_type=4,
                 embedding__vocab_position=1000,
                 embedding__pad_idx_value=2,
                 embedding__pad_idx_coord_type=0,
                 embedding__pad_idx_position=0,
                 reformer__depth=12,
                 reformer__heads=8,
                 reformer__n_hashes=8,
                 reformer__bucket_size=48,
                 reformer__causal=True,
                 reformer__lsh_dropout=0.2,
                 reformer__ff_dropout=0.2,
                 reformer__post_attn_dropout=0.2,
                 reformer__ff_mult=4):

        # dataset config
        train_dataset_config = {
            "root_dir":  r"/mnt/users/ikostiuk/local/PolyGen/polygen_exports",
            "classes":   ['02691156'],
            "transform": [SortVertices(),
                          NormalizeVertices(),
                          QuantizeVertices(),
                          ToTensor(),
                          ResizeVertices(799)
                          ],
            "split": 'train',
            "train_percentage": 0.9

        }

        # tokenizer config
        tokenizer_config = {
            "bos_id": tokenizer__bos_id,
            "eos_id": tokenizer__eos_id,
            "pad_id": tokenizer__pad_id,
            "max_seq_len": max_seq_len,
        }

        # embedding config
        embedding_config = {
            "vocab_value": embedding__vocab_value,
            "vocab_coord_type": embedding__vocab_coord_type,
            "vocab_position": embedding__vocab_position,
            "pad_idx_value": embedding__pad_idx_value,
            "pad_idx_coord_type": embedding__pad_idx_coord_type,
            "pad_idx_position": embedding__pad_idx_position,
            "embed_dim": embed_dim,
        }

        # reformer info
        reformer_config = {
            "dim": embed_dim,
            "depth": reformer__depth,
            "max_seq_len": max_seq_len,
            "heads": reformer__heads,
            "bucket_size": reformer__bucket_size,
            "n_hashes": reformer__n_hashes,
            "causal": reformer__causal,
            "lsh_dropout": reformer__lsh_dropout,
            "ff_dropout": reformer__ff_dropout,
            "post_attn_dropout": reformer__post_attn_dropout,
            "ff_mult": reformer__ff_mult,
        }

        self.config = {
            "train_dataset": train_dataset_config,
            "embed_dim": embed_dim,
            "max_seq_len": max_seq_len,
            "tokenizer": tokenizer_config,
            "embedding": embedding_config,
            "reformer": reformer_config,
        }
