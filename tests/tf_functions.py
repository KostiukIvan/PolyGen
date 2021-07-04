import numpy as np
import tensorflow as tf


def normalize_vertices_scale(vertices):
  """Scale the vertices so that the long diagonal of the bounding box is one."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  scale = np.sqrt(np.sum(extents**2))
  return vertices / scale


def quantize_verts(verts, n_bits=8):
  """Dequantizes integer vertices to floats."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts_quantize = (
      (verts - min_range) * range_quantize / (max_range - min_range))
  return tf.cast(verts_quantize, tf.int32)


def dequantize_verts(verts, n_bits, add_noise=False):
  """Quantizes vertices and outputs integers with specified n_bits."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts = tf.cast(verts, tf.float32)
  verts = verts * (max_range - min_range) / range_quantize + min_range
  if add_noise:
    verts += tf.random_uniform(tf.shape(verts)) * (1 / float(range_quantize))
  return verts


def detokenize(vertices, num_vertices, num_samples, quantization_bits):
  pass
