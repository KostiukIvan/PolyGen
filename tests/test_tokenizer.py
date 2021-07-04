"""
Libraries needed for tests:
 - 'tensorflow==1.5'
 - 'torch==1.6' (higher version should work as well)
 - 'numpy'
"""


import unittest

import torch
import tensorflow as tf
import numpy as np

import tf_functions
from data_utils.transformations import VertexTokenizer, QuantizeVertices, ToTensor, NormalizeVertices, SortVertices


class TestTokenizer(unittest.TestCase):
    """
    Test case which checks if tokenization & detokenization
    works as in tf1.x implementation.
    """

    def setUp(self):

        self.quantize_vertices = QuantizeVertices()
        self.to_tensor = ToTensor()
        self.sort_vertices = SortVertices()  # We used identical method to sort vertices [no need to test]
        self.normalize = NormalizeVertices()

        self.max_seq_len = 99
        self.vertex_tokenizer = VertexTokenizer(self.max_seq_len)

    def test_normalize(self):
        random_vertices = self.sort_vertices(np.random.uniform(-50, 50, size=(10, 3)))
        torch_normalized = self.normalize(random_vertices)
        tf_normalized = tf_functions.normalize_vertices_scale(random_vertices)

        print(random_vertices, torch_normalized, tf_normalized, sep='\n')
        self.assertTrue(np.allclose(torch_normalized, tf_normalized))

    def test_quantize(self):
        normalized_vertices = self.sort_vertices(np.random.uniform(0, 1, size=(10, 3)))

        torch_quantized = self.quantize_vertices(normalized_vertices)
        with tf.Session() as sess:
            tf_quantized = tf_functions.quantize_verts(normalized_vertices).eval()

        print(normalized_vertices, torch_quantized, tf_quantized, sep='\n')
        self.assertTrue(np.allclose(torch_quantized, tf_quantized))

    def test_tokenize(self):
        quantized_vertices = self.sort_vertices(
            np.random.randint(0, 256, (10, 3))
        ).flatten()
        len_verts = len(quantized_vertices)

        torch_position_tokens = torch.arange(len_verts) % 3
        torch_cord_tokens = torch.arange(len_verts) // 3
        with tf.Session() as sess:
            tf_position_tokens = tf.mod(tf.range(len_verts), 3).eval()
            tf_cord_tokens = tf.floordiv(tf.range(len_verts), 3).eval()

        print(torch_position_tokens, tf_position_tokens, sep='\n')
        print(torch_cord_tokens, tf_cord_tokens, sep='\n')
        self.assertTrue(np.allclose(torch_position_tokens.numpy(), tf_position_tokens))
        self.assertTrue(np.allclose(torch_cord_tokens.numpy(), tf_cord_tokens))


if __name__ == "__main__":
    unittest.main()
