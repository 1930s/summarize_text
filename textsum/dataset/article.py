# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re, string

from nltk.tokenize import word_tokenize

import numpy as np

import tensorflow as tf


def truncate_right_and_pad_list(l, max_len=1, pad_with=0):
  """Truncates a list from 0:`max_len` and pads with 0 if len(l) < max_len.

  Returns:
    fixed length list.

  """
  l_len = len(l)
  if l_len > max_len:
    l = l[:max_len]
  elif l_len < max_len:
    l += [pad_with] * (max_len - l_len)
  return l


def clean_text(text):
  text = validate_text(text)
  text = text.replace('…', '...')                             # replace elipses with 3 periods.
  text = re.sub(r'\s+', ' ', text)                            # replace multiple spaces with single.
  text = text.replace('“', '"')                               # replace unique double quotes
  text = text.replace('”', '"')                               # replace unique double quotes
  text = text.replace('‘', "'")
  text = text.replace(',', ',')
  text = text.translate(str.maketrans('', '', string.digits)) # remove digits.
  text = re.sub(r'http\S+', ' ', text)                        # remove links
  text = text.replace("'", '')
  text = text.replace('	', '')
  text = text.replace('`', '')
  return text

def tokenize(text):
  text = clean_text(text)
  text = text.replace('\n', ' ').replace('\t', ' ')
  text = ' '.join(text.split())
  text = word_tokenize(text.lower())
  return text


def validate_text(text):
  """If text is not str or unicode, then try to convert it to str."""
  return str(text)


class Article(object):
  # for validation purposes only ¯\_(ツ)_/¯
  max_topic = 1

  # "When you publish a new post, you can add up to five tags."
  # Reference: https://help.medium.com/hc/en-us/articles/214741038-Tags
  max_tags = 5

  # "The ideal length of a blog post is 7 minutes, 1,600 words"
  # Reference: https://blog.bufferapp.com/the-ideal-length-of-everything-online-according-to-science
  max_text = 750

  # this is all hypothetical...
  max_title = 11
  max_subtitle = max_title
  max_short_description = 10
  max_description = 50

  max_lookup = {
    'topic': max_topic+2,
    'text': max_text+2,
    'title': max_title+2,
    'subtitle': max_subtitle+2,
    'tags': max_tags+2,
    'short_description': max_short_description+2,
    'description': max_description+2,
  }

  SPACE = '<S>'

  # pad token because every vector (list of words) has to be same size.
  PAD = '<P>'
  # end of sequence token to signal stopping with or without padding
  EOS = '<E>'
  # go token signals start of sequence.
  GO = '<G>'

  json_keys = ['title', 'subtitle', 'text', 'tags', 'shortDescription']
  tf_keys = ['title', 'subtitle', 'text', 'tags', 'short_description', 'description']

  def __init__(self,
               topic='',
               title='',
               subtitle='',
               text='',
               description='',
               tags=[],
               shortDescription=''):
    """Create an Article.

    Note:
      features are 'right' padded.

    """
    self.topic = [validate_text(topic)]
    self.title = tokenize(title)
    self.subtitle = tokenize(subtitle)
    self.text = tokenize(text)
    self.tags = [validate_text(tag) for tag in tags]
    self.description = tokenize(description)
    self.short_description = tokenize(shortDescription)

    self.feature_vocabularies = {
      'topic': set(self.topic),
      'title': set(self.title),
      'subtitle': set(self.subtitle),
      'text': set(self.text),
      'tags': set(self.tags),
      'short_description': set(self.short_description),
      'description': set(self.description),
    }

  def to_tf_example(self, pad=True):
    """Serializes article to a `tf.train.Example` for consumption of `tf.Record*`

    Returns:
      a `tf.train.Example` representing the Article.

    """
    if pad:
      title = [Article.GO] + truncate_right_and_pad_list(self.title,
        max_len=Article.max_title,
        pad_with=Article.PAD) + [Article.EOS]
      subtitle = [Article.GO] + truncate_right_and_pad_list(self.subtitle,
        max_len=Article.max_subtitle,
        pad_with=Article.PAD) + [Article.EOS]
      text = [Article.GO] + truncate_right_and_pad_list(self.text,
        max_len=Article.max_text,
        pad_with=Article.PAD) + [Article.EOS]
      tags = [Article.GO] + truncate_right_and_pad_list(self.tags,
        max_len=Article.max_tags,
        pad_with=Article.PAD) + [Article.EOS]
      short_description = [Article.GO] + truncate_right_and_pad_list(self.short_description,
        max_len=Article.max_short_description,
        pad_with=Article.PAD) + [Article.EOS]
      description = [Article.GO] + truncate_right_and_pad_list(self.short_description,
        max_len=Article.max_description,
        pad_with=Article.PAD) + [Article.EOS]
    else:
      title = [Article.GO] + self.title + [Article.EOS]
      subtitle = [Article.GO] + self.subtitle + [Article.EOS]
      text = [Article.GO] + self.text + [Article.EOS]
      tags = [Article.GO] + self.tags + [Article.EOS]
      short_description = [Article.GO] + self.short_description + [Article.EOS]
      description = [Article.GO] + self.description + [Article.EOS]
    topic = [Article.GO] + self.topic + [Article.EOS]

    features = {
      'topic': topic,
      'title': title,
      'subtitle': subtitle,
      'text': text,
      'tags': tags,
      'short_description': short_description,
      'description': description,
    }

    sequence_example = tf.train.Example()
    for k, v in features.items():
      sequence_example.features.feature['{}_length'.format(k)].int64_list.value.append(len(v))
      feature = sequence_example.features.feature[k].bytes_list.value
      for seq_v in v:
        feature.append(tf.compat.as_bytes(seq_v))
    return sequence_example

  def to_numpy(self, pad=False):
    """Serializes article to a dictionary of numpy variables.

    Returns:
      dict of numpy features.

    """
    topic_length = len(self.topic) + 2
    title_length = len(self.title) + 2
    subtitle_length = len(self.subtitle) + 2
    text_length = len(self.text) + 2
    tags_length = len(self.tags) + 2
    short_description_length = len(self.short_description) + 2
    description_length = len(self.description) + 2
    if pad:
      title = [Article.GO] + truncate_right_and_pad_list(self.title,
        max_len=Article.max_title,
        pad_with=Article.PAD) + [Article.EOS]
      subtitle = [Article.GO] + truncate_right_and_pad_list(self.subtitle,
        max_len=Article.max_subtitle,
        pad_with=Article.PAD) + [Article.EOS]
      text = [Article.GO] + truncate_right_and_pad_list(self.text,
        max_len=Article.max_text,
        pad_with=Article.PAD) + [Article.EOS]
      tags = [Article.GO] + truncate_right_and_pad_list(self.tags,
        max_len=Article.max_tags,
        pad_with=Article.PAD) + [Article.EOS]
      short_description = [Article.GO] + truncate_right_and_pad_list(self.short_description,
        max_len=Article.max_short_description,
        pad_with=Article.PAD) + [Article.EOS]
      description = [Article.GO] + truncate_right_and_pad_list(self.description,
        max_len=Article.max_description,
        pad_with=Article.PAD) + [Article.EOS]
    else:
      title = [Article.GO] + self.title + [Article.EOS]
      subtitle = [Article.GO] + self.subtitle + [Article.EOS]
      text = [Article.GO] + self.text + [Article.EOS]
      tags = [Article.GO] + self.tags + [Article.EOS]
      short_description = [Article.GO] + self.short_description + [Article.EOS]
      description = [Article.GO] + self.description + [Article.EOS]
    topic = [Article.GO] + self.topic + [Article.EOS]

    features = {
      'topic': np.array([tf.compat.as_bytes(s) for s in topic], dtype=bytes),
      'topic_length': np.array([topic_length], dtype=np.int64),
      'title': np.array([tf.compat.as_bytes(s) for s in title], dtype=bytes),
      'title_length': np.array([title_length], dtype=np.int64),
      'subtitle': np.array([tf.compat.as_bytes(s) for s in subtitle], dtype=bytes),
      'subtitle_length': np.array([subtitle_length], dtype=np.int64),
      'text': np.array([tf.compat.as_bytes(s) for s in text], dtype=bytes),
      'text_length': np.array([text_length], dtype=np.int64),
      'tags': np.array([tf.compat.as_bytes(s) for s in tags], dtype=bytes),
      'tags_length': np.array([tags_length], dtype=np.int64),
      'short_description': np.array([tf.compat.as_bytes(s) for s in short_description], dtype=bytes),
      'short_description_length': np.array([short_description_length], dtype=np.int64),
      'description': np.array([tf.compat.as_bytes(s) for s in description], dtype=bytes),
      'description_length': np.array([description_length], dtype=np.int64),
    }

    return features
