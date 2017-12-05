# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import tensorflow as tf


def normalized_columns_initializer(stdv=1.0):
  """factory for normalizing over columns given stdv."""
  def initializer(shape, dtype=None, partition_info=None):
    out = np.random.randn(*shape).astype(np.float32)
    out *= stdv / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)
  return initializer


def create_article_model(features, labels, graph,
                         vocab_sizes=(),
                         mode=None,
                         hparams=None,
                         seed=0):
  """Factory for creating the article model."""
  model = ArticleModel(features, labels, graph,
                       vocab_sizes=vocab_sizes,
                       mode=mode,
                       hparams=hparams,
                       seed=seed)
  return model.prediction_op, model.loss_op, model.train_op


class ArticleModel(object):

  def __init__(self, features, labels, graph,
               vocab_sizes=(),
               mode=None,
               hparams=None,
               seed=0):
    """Sequence to sequence model."""
    self._mode = mode
    self._features = features
    self._labels = labels
    self._graph = graph
    self._vocab_sizes = vocab_sizes
    self._hparams = hparams
    self._seed = seed

    self.seq2seq()

  @property
  def prediction_op(self):
    return self._prediction_op

  @property
  def loss_op(self):
    return self._loss_op

  @property
  def train_op(self):
    return self._train_op

  def _select_cell(self, cell_type):
    if cell_type == 'basic_lstm': return tf.nn.rnn_cell.BasicLSTMCell
    elif cell_type == 'basic_rnn': return tf.nn.rnn_cell.BasicRNNCell
    elif cell_type == 'gru': return tf.nn.rnn_cell.GRUCell
    elif cell_type == 'lstm': return tf.nn.rnn_cell.LSTMCell
    elif cell_type == 'multi_rnn': return tf.nn.rnn_cell.MultiRNNCell
    elif cell_type == 'rnn': return tf.nn.rnn_cell.RNNCell
    else: raise ValueError('{} cell type not supported'.format(cell_type))

  def _select_attention(self, attention_type):
    if attention_type == 'luong': return tf.contrib.seq2seq.LuongAttention
    elif attention_type == 'luong_monotonic': return tf.contrib.seq2seq.LuongMonotonicAttention
    elif attention_type == 'bahdanau_monotonic': return tf.contrib.seq2seq.BahdanauMonotonicAttention
    elif attention_type == 'bahdanau': return tf.contrib.seq2seq.BahdanauAttention
    else: raise ValueError('{} attention type not supported'.format(attention_type))

  def _create_encoder(self, input_embed, input_sequence_length):
    cell = lambda: self._select_cell(self._hparams.input_encoder_cell_type)(self._hparams.input_n_encoder_units)
    if self._hparams.use_bidirectional:
      encoder_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
          [cell() for _ in range(self._hparams.input_n_encoder_layers)],
          [cell() for _ in range(self._hparams.input_n_encoder_layers)],
          input_embed,
          dtype=tf.float32,
          sequence_length=input_sequence_length,
          time_major=False)
      return encoder_outputs, None
    else:
      cell = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self._hparams.input_n_encoder_layers)],
                                         state_is_tuple=True)
      return tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)

  def seq2seq(self):
    with self._graph.as_default():
      input_vocab_size, target_vocab_size = self._vocab_sizes

      input_sequence_length = tf.squeeze(self._features['input_sequence_length'], [-1])
      target_sequence_length = tf.squeeze(self._labels['target_sequence_length'], [-1])

      start_tokens = tf.zeros([self._hparams.batch_size], dtype=tf.int64)
      train_output = tf.concat([tf.expand_dims(start_tokens, 1), self._labels['target']], 1)

      input_embeddings = tf.get_variable('input_embeddings', [input_vocab_size, self._hparams.input_embedding_size],
                                         initializer=normalized_columns_initializer())

      input_embed = tf.nn.embedding_lookup(params=input_embeddings,
                                           ids=self._features['input'])

      target_embeddings = tf.get_variable('target_embeddings', [target_vocab_size, self._hparams.target_embedding_size],
                                          initializer=normalized_columns_initializer())
      target_embed = tf.nn.embedding_lookup(params=target_embeddings,
                                            ids=self._labels['target'])

      encoder_outputs, encoder_final_state = self._create_encoder(input_embed, input_sequence_length)
      train_helper = tf.contrib.seq2seq.TrainingHelper(target_embed, target_sequence_length)

      pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
          target_embeddings,
          start_tokens=tf.to_int32(start_tokens),
          end_token=2)

      def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
          cell = lambda: self._select_cell(self._hparams.target_decoder_cell_type)(self._hparams.target_n_decoder_units)
          cell = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self._hparams.target_n_decoder_layers)],
                                             state_is_tuple=True)
          attention_mechanism = self._select_attention(self._hparams.attention_mechanism)(
              num_units=self._hparams.target_n_decoder_units,
              memory=encoder_outputs,
              memory_sequence_length=input_sequence_length)
          attn_cell = tf.contrib.seq2seq.AttentionWrapper(
              cell, attention_mechanism,
              attention_layer_size=int(self._hparams.target_n_decoder_units / 2))
          out_cell = tf.contrib.rnn.OutputProjectionWrapper(
              attn_cell, target_vocab_size,
              reuse=reuse,
          )
          decoder = tf.contrib.seq2seq.BasicDecoder(
              cell=out_cell, helper=helper,
              initial_state=out_cell.zero_state(
                  dtype=tf.float32, batch_size=self._hparams.batch_size))
          outputs = tf.contrib.seq2seq.dynamic_decode(
              decoder=decoder,
              output_time_major=False,
              impute_finished=True,
              maximum_iterations=tf.reduce_max(target_sequence_length) * 2
          )
          return outputs[0]

      train_outputs = decode(train_helper, 'decode')
      pred_outputs = decode(pred_helper, 'decode', reuse=True)

      tf.identity(train_outputs.sample_id[0], name='train_pred')
      weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
      loss = tf.contrib.seq2seq.sequence_loss(
          train_outputs.rnn_output, self._labels['target'],
          weights=weights)
      train_op = tf.contrib.layers.optimize_loss(
          loss, tf.train.get_global_step(),
          optimizer=self._hparams.optimizer,
          learning_rate=self._hparams.learning_rate,
          summaries=['loss', 'learning_rate'])

      self._prediction_op = pred_outputs
      self._loss_op = loss
      self._train_op = train_op
