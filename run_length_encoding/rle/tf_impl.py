"""Tensorflow encode/decode/utility implementations for run length encodings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_datasets.core import tf_compat


def brle_length(brle):
  """Length of dense form of binary run-length-encoded input.

  Efficient implementation of `len(brle_to_dense(brle))`."""
  return tf.reduce_sum(brle)


def rle_length(rle):
  """Length of dense form of run-length-encoded input.

  Efficient implementation of `len(rle_to_dense(rle))`."""
  return tf.reduce_sum(rle[1::2])


def brle_logical_not(brle):
  """Get the BRLE encoding of the `logical_not`ed decoding of brle.

  Equivalent to `dense_to_brle(tf.logical_not(brle_to_dense(brle)))` but highly
  optimized. Actual implementation just pads brle with a 0 on each end.

  Args:
    brle: rank 1 int tensor of ints.
  """
  return tf.pad(brle, [[1, 1]])


def brle_to_dense(brle, vals=None):
  """Decode binary run length encoded data.

  Args:
    brle: binary run length encoded data, 1D tensor of int dtype
    vals (optional): length-2 tensor/array/tuple made up of
        (false_val, true_val).

  Returns:
    Dense representation, rank-1 tensor with dtype the same as vals or `tf.bool`
      if `vals is None`.
  """
  if hasattr(tf, 'RaggedTensor'):
    return _brle_to_dense_ragged(brle, vals)
  else:
    return _brle_to_dense_repeat(brle, vals)


def _brle_to_dense_repeat(brle, vals=None):
  if vals is None:
    vals = tf.consatnt([False, True], dtype=tf.bool)
  else:
    vals = tf.convert_to_tensor(vals)
    if vals.shape != (2,):
      raise ValueError(
        'vals must be a 2 element tensor, got shape %s' % (vals.shape))
  if brle.shape.ndims != 1:
    raise ValueError(
      'brle must be a rank 1 tensor, got shape %s' % (brle.shape))
  vals = tf.tile(vals, tf.size(brle) // 2)
  return tf.repeat(vals, brle)


def _brle_to_dense_ragged(brle, vals=None):
  """brle_to_dense implementation using RaggedTensor. Requires tf 1.13."""
  if not hasattr(tf, 'RaggedTensor'):
    raise RuntimeError('_brle_to_dense_ragged requires tf.RaggedTensor')

  brle = tf.convert_to_tensor(brle)
  if not brle.dtype.is_integer:
    raise TypeError("`brle` must be of integer type.")
  if not brle.shape.ndims == 1:
    raise TypeError("`brle` must be a rank 1 tensor.")
  enc_false, enc_true = tf.unstack(tf.reshape(brle, (-1, 2)), axis=1)

  n_false = tf.reduce_sum(enc_false)
  n_true = tf.reduce_sum(enc_true)
  if vals is None:
    false_vals = tf.zeros((n_false,), dtype=tf.bool)
    true_vals = tf.ones((n_true,), dtype=tf.bool)
  else:
    vals = tf.convert_to_tensor(vals)
    if vals.shape.ndims != 1:
      raise ValueError("`vals` must be rank 1 tensor.")
    if vals.shape[0] != 2:
      raise ValueError("`vals` must have exactly 2 entries.")

    false_val, true_val = tf.unstack(vals, axis=0)
    false_vals = tf.fill([n_false], false_val)
    true_vals = tf.fill([n_true], true_val)

  ragged_false = tf.RaggedTensor.from_row_lengths(false_vals, enc_false)
  ragged_true = tf.RaggedTensor.from_row_lengths(true_vals, enc_true)

  ragged_all = tf.concat([ragged_false, ragged_true], axis=1)
  return ragged_all.values


def tf_repeat(values, repeats):
  values = tf.convert_to_tensor(values)
  repeats = tf.convert_to_tensor(repeats)

  if values.shape.ndims != 1:
    raise ValueError('values must be rank 1, got shape %s' % values.shape)
  if repeats.shape.ndims != 1:
    raise ValueError('repeats must be rank 1, got shape %s' % repeats.shape)
  if not repeats.dtype.is_integer:
    raise ValueError('repeats must be an integer, got %s' % repeats.dtype)

  def fold_fn(red, elems):
    value, repeat = elems
    filled = tf.fill((repeat,), value)
    return tf.concat((red, filled), axis=0)

  out = tf.foldl(
      fold_fn, (values, repeats),
      back_prop=False, initializer=tf.zeros(shape=(0,), dtype=values.dtype))

  return out


def rle_to_dense(rle, dtype=tf.int64):
  """Convert run length encoded data to dense.

  Note current implementation uses `tf.py_function` in `tf_repeat`.
  """
  rle = tf.convert_to_tensor(rle)
  values, counts = tf.unstack(tf.reshape(rle, (-1, 2)), axis=-1)
  return tf_repeat(values, counts)
