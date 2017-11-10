# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json, os, string, struct, time

import tensorflow as tf

import numpy as np

from textsum.processing import ThreadPool
from .article import Article


class TFRecordStream(object):
  def __init__(self, src, max_threads=8):
    """Stream the `src` directory and process files using multiple threads. This assumes that src is
       already sharded. This might be changed in the future ¯\_(ツ)_/¯.

    Args:
      src:
        readable directory.
      max_threads:
        maximum concurrent threads at a time.

    """
    self._src_directory = src
    self._thread_pool = ThreadPool(max_threads)
    self._article_hash_set = set()
    self._article_hash_key = 'title'

  def _is_valid_article(self, article):
    for key in Article.json_keys:
      if key in article:
        dummy = ' '.join(article[key])
        if ''.join(dummy.split()) is '':
          print(key, 'empty')
          return False
      else:
        print(key, 'empty')
        return False
    return True

  def _new_article_vocabulary(self):
    return {k: set() for k in Article.tf_keys}

  def _write_article(self, idx, name, article, feature_vocabularies, pad):
    record_filename = name + '_' + str(idx) + '.npy'

    # validate against the schema.
    if not self._is_valid_article(article):
      print('skipping article')
      return 0

    with self._thread_pool.lock:
      # hash the unique article, skip if already encountered.
      article_hash = hash(article[self._article_hash_key])
      if article_hash in self._article_hash_set:
        print('found duplicate article: {}'.format(article_hash))
        return 0

    self._article_hash_set.add(article_hash)

    if os.path.isfile(record_filename):
      print(record_filename)
      print('record written already.')
      return 0

    wrapped = Article(**article)
    vocab = wrapped.feature_vocabularies

    for k in feature_vocabularies.keys():
      feature_vocabularies[k] = feature_vocabularies[k].union(vocab[k])

    np.save(record_filename, wrapped.to_numpy(pad=pad))
    return 1

  def _read_articles(self, name='', filename='', ext='.json', pad=False):
    with tf.gfile.Open(filename, 'r') as df:
      articles = json.loads(df.read())
      wrapped_articles = []
      feature_vocabularies = self._new_article_vocabulary()
      idx = 0
      for article in articles:
        idx += self._write_article(idx, name, article, feature_vocabularies, pad)
      return feature_vocabularies

  def _write_vocab(self, filename, vocab_set):
    with tf.gfile.Open(filename, 'w') as f:
      f.write('{}\n'.format('<G>'))
      f.write('{}\n'.format('<P>'))
      f.write('{}\n'.format('<E>'))
      f.write('{}\n'.format('<U>'))
      for v in vocab_set:
        f.write('{}\n'.format(v))

  def pipe(self, dst, pad=False):
    """Consume the `src` into `directory` as `tf.Records`.

    Args:
      directory:
        writable directory.
      pad:
        pad sequences to their max lengths.

    """
    try: os.makedirs(dst)
    except: pass
    all_files = os.listdir(self._src_directory)

    feature_vocabularies = self._new_article_vocabulary()

    for data_fn in all_files:
      name, ext = os.path.splitext(data_fn)
      self._thread_pool.add_task(self._read_articles,
        name=os.path.join(dst, name),
        filename=os.path.join(self._src_directory, data_fn),
        ext=ext,
        pad=pad)

    for worker in self._thread_pool.wait_completion():
      for vocab in worker:
        for k in feature_vocabularies.keys():
          feature_vocabularies[k] = feature_vocabularies[k].union(vocab[k])

    for k, v in feature_vocabularies.items():
      self._write_vocab(os.path.join(dst, '{}_vocab.txt'.format(k)), v)
