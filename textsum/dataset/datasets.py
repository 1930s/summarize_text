# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import numpy as np

import tensorflow as tf

from textsum.dataset import Article


def create_article_dataset(record_list, dataset_dir, sess,
                           validation_size=10,
                           eval_every=100,
                           input_feature='text',
                           max_input_sequence_length=Article.max_text+2,
                           target_feature='short_description',
                           max_target_sequence_length=Article.max_short_description+2,
                           hparams=None):
  """Creates a dataset of records from articles.

  Args:
    record_list:
      placeholder of list of articles.
    target_feature:
      which feature to target.
    input_feature:
      which feature to input.
    hparams:
      hyper parameters.

  """
  input_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=os.path.join(dataset_dir, '{}_vocab.txt'.format(input_feature)),
                                                        num_oov_buckets=1,
                                                        default_value=3)
  target_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=os.path.join(dataset_dir, '{}_vocab.txt'.format(target_feature)),
                                                         num_oov_buckets=1,
                                                         default_value=3)
  lookup_table = tf.contrib.lookup.index_to_string_table_from_file(
    os.path.join(dataset_dir, '{}_vocab.txt'.format(target_feature)),
    default_value='<U>')

  input_lookup_table = tf.contrib.lookup.index_to_string_table_from_file(
    os.path.join(dataset_dir, '{}_vocab.txt'.format(input_feature)),
    default_value='<U>')

  tf.tables_initializer().run(session=sess)

  dataset = tf.data.Dataset.from_tensor_slices(record_list)

  def _read_npy(filename):
    parsed = np.load(filename.decode('utf-8'))
    data = parsed.item()
    i = list(map(lambda x: x.decode('utf-8'), data[input_feature].tolist()))
    i_l = data['{}_length'.format(input_feature)]
    t = list(map(lambda x: x.decode('utf-8'), data[target_feature].tolist()))
    t_l = data['{}_length'.format(target_feature)]
    return i, i_l, t, t_l

  def next_example(input_feature, input_sequence_length, target_feature, target_sequence_length):
    input_sequence_length.set_shape([1])
    input_feature.set_shape([max_input_sequence_length])
    target_sequence_length.set_shape([1])
    target_feature.set_shape([max_target_sequence_length])

    feature_input_sequences = {
      'input_sequence_length': tf.cast([max_input_sequence_length], tf.int32),
    }
    feature_target_sequences = {
      'target_sequence_length': tf.cast([max_target_sequence_length], tf.int32),
    }

    feature_target_sequences['target'] = target_table.lookup(target_feature)
    feature_input_sequences['input'] = input_table.lookup(input_feature)
    return feature_input_sequences, feature_target_sequences

  dataset = dataset.map(lambda filename: tf.py_func(_read_npy, [filename], [tf.string, tf.int64, tf.string, tf.int64]))
  dataset = dataset.map(next_example)
  dataset = dataset.shuffle(buffer_size=hparams.shuffle_buffer_size)

  def training_set(dataset):
    iterator = dataset.make_initializable_iterator()
    iterator.initializer.run(session=sess)
    def train():
      return iterator.get_next()
    return train

  def validation_set(dataset):
    iterator = dataset.make_initializable_iterator()
    iterator.initializer.run(session=sess)
    def validate():
      return iterator.get_next()
    return validate

  dataset = dataset.batch(hparams.batch_size)
  validation_dataset = dataset.take(validation_size)
  training_dataset = dataset.repeat(hparams.epochs)
  training_dataset = training_dataset.shuffle(buffer_size=hparams.shuffle_buffer_size)
  train = training_set(training_dataset)
  valid = validation_set(validation_dataset)
  input_vocab_size = input_table.size().eval(session=sess)
  target_vocab_size = target_table.size().eval(session=sess)

  return train, valid, (input_vocab_size, target_vocab_size), lookup_table, input_lookup_table


def run_article_experiment(model, hparams,
                           mode=tf.contrib.learn.ModeKeys.TRAIN,
                           validation_size=10,
                           input_feature='text',
                           max_input_sequence_length=Article.max_text+2,
                           target_feature='short_description',
                           max_target_sequence_length=Article.max_short_description+2,
                           dataset_dir='records/medium',
                           model_dir='model',
                           seed=0,
                           restore=True,
                           just_evaluate_me=True):
  """Creates a set of `tf.contrib.learn` API tools for running an experiment on a model.

  Args:
    model:
      returns a tuple containing (prediction_op, loss_op, train_op)
    hparams:
      model and dataset hyperparameters.
    mode:
      experiment schedule.
    target_feature:
      which feature to target.
    input_feature:
      which feature to input.
    dataset_dir:
      where the dataset is.

  Returns:
    the estimator to run.

  """
  eval_every = 100

  graph = tf.get_default_graph()
  sess = tf.Session(graph=graph)
  run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)
  all_files = os.listdir(dataset_dir)
  record_list = []
  for fn in all_files:
    if fn.endswith('.txt'):
      continue
    record_list.append(os.path.join(dataset_dir, fn))
  train_input_fn, eval_input_fn, vocab_sizes, lookup_table, input_lookup_table = create_article_dataset(record_list, dataset_dir, sess,
                                                                                                        validation_size=validation_size,
                                                                                                        eval_every=eval_every,
                                                                                                        input_feature=input_feature,
                                                                                                        target_feature=target_feature,
                                                                                                        max_input_sequence_length=max_input_sequence_length,
                                                                                                        max_target_sequence_length=max_target_sequence_length,
                                                                                                        hparams=hparams)

  train_features, train_labels = train_input_fn()
  # eval_features, eval_labels = eval_input_fn()
  prediction_op, loss_op, train_op = model(train_features, train_labels, graph,
                                           mode=mode,
                                           vocab_sizes=vocab_sizes,
                                           hparams=hparams,
                                           seed=seed)

  # actual inference.
  input_strings = input_lookup_table.lookup(train_features['input'])
  pred_strings = lookup_table.lookup(tf.cast(prediction_op.sample_id, dtype=tf.int64))
  label_strings = lookup_table.lookup(train_labels['target'])

  # inputs_text = tf.summary.text('inputs', input_strings)
  # predictions_text = tf.summary.text('predictions', pred_strings)
  # targets_text = tf.summary.text('targets', label_strings)
  loss_summary = tf.summary.scalar('loss', loss_op)
  # merged_text = tf.summary.merge([inputs_text, predictions_text, targets_text])


  saver = tf.train.Saver()
  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(os.path.join(model_dir, 'log'), sess.graph)
  if restore:
    saver.restore(sess, os.path.join(model_dir, 'model_restored.ckpt'))

  if just_evaluate_me:
    while True:
      try:
        batch_train_features, batch_train_labels = sess.run((train_features, train_labels))
        inputs_, preds_, labels_ = sess.run([input_strings, pred_strings, label_strings], feed_dict={
          train_features['input']: batch_train_features['input'],
          train_features['input_sequence_length']: batch_train_features['input_sequence_length'],
          train_labels['target']: batch_train_labels['target'],
          train_labels['target_sequence_length']: batch_train_labels['target_sequence_length'],
        })
        for inputs, pred, label in zip(inputs_, preds_, labels_):
          print('Input')
          print('='*30)
          print(' '.join(list(map(lambda s: s.decode('utf-8'), inputs))))
          print('Actual')
          print('='*30)
          print(' '.join(list(map(lambda s: s.decode('utf-8'), label))))
          print('Prediction')
          print('='*30)
          print(' '.join(list(map(lambda s: s.decode('utf-8'), pred))))
          print('\n'*2)
      except tf.errors.OutOfRangeError:
        break
    sess.close()

  else:
    epoch = 0
    save_every = 100
    while True:
      try:
        batch_train_features, batch_train_labels = sess.run((train_features, train_labels))
        if batch_train_features['input'].shape[0] != hparams.batch_size:
          continue
        if batch_train_labels['target'].shape[0] != hparams.batch_size:
          continue
        if batch_train_labels['target_sequence_length'].shape[0] != hparams.batch_size:
          continue
        if batch_train_features['input_sequence_length'].shape[0] != hparams.batch_size:
          continue

        summ_loss, _, loss = sess.run([loss_summary, train_op, loss_op], feed_dict={
          train_features['input']: batch_train_features['input'],
          train_features['input_sequence_length']: batch_train_features['input_sequence_length'],
          train_labels['target']: batch_train_labels['target'],
          train_labels['target_sequence_length']: batch_train_labels['target_sequence_length'],
        })

        writer.add_summary(summ_loss, epoch)

        print('loss at epoch {}: {}'.format(epoch, loss))

        if (epoch + 1) % save_every == 0:
          saver.save(sess, os.path.join(model_dir, 'model_restored.ckpt'))

        # if (epoch + 1) % eval_every == 0:
        #   features, labels = sess.run((eval_features, eval_labels))
        #
        #   if features['input'].shape[0] != hparams.batch_size:
        #     continue
        #   if labels['target'].shape[0] != hparams.batch_size:
        #     continue
        #   if labels['target_sequence_length'].shape[0] != hparams.batch_size:
        #     continue
        #   if features['input_sequence_length'].shape[0] != hparams.batch_size:
        #     continue
        #
        #   text_summary, _, _ = sess.run([merged_text, pred_strings, label_strings], feed_dict={
        #     eval_features['input']: features['input'],
        #     eval_features['input_sequence_length']: features['input_sequence_length'],
        #     eval_labels['target']: labels['target'],
        #     eval_labels['target_sequence_length']: labels['target_sequence_length'],
        #   })
        #   writer.add_summary(text_summary, epoch)

        epoch += 1
      except tf.errors.OutOfRangeError:
        break
    sess.close()
