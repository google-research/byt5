# Copyright 2023 The ByT5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics for evaluating Byte models."""

import re
import string

import nltk
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from t5.evaluation import metrics as t5_metrics


def cer(targets, predictions):
  """Computes the Character Error Rate (CER).

  The Character Error Rate for a (input word, target word) pair is defined as
  the minimum number of edits required to transform the input word to the target
  word divided by the total number of characters in the target word. The minimum
  number of edits is calculated using Levenshtein distance, where any
  single-character edit (insertion, deletion, substitution) is allowed.
  The CER for a list of (input word, target word) pairs is defined as the total
  number of edits required to transform each input word into the corresponding
  target word divided by the sum of the number of characters in the target
  words. For example, given:
    targets = ["abc", "aa"]
    predictions = ["abd", "a"]
  the CER would be: (1 + 1) / (3 + 2) = 2 / 5 = 0.4

  Args:
    targets: list of strings.
    predictions: list of strings.

  Returns:
    float, CER value for the predictions compared to the targets.
  """
  total_characters = 0
  total_edit_distance = 0
  for target, prediction in zip(targets, predictions):
    total_edit_distance += nltk.edit_distance(target, prediction)
    total_characters += len(target)

  return {'cer': float(total_edit_distance) / total_characters}


def _normalize_text(text):
  """Normalize text for TweetQA task.

  Args:
    text: string
  Returns:
    normalized string
  """

  # Lower case.
  text = text.lower()

  # Remove punctuation.
  text = ''.join(ch for ch in text if ch not in set(string.punctuation))

  # Remove articles.
  text = re.sub(r'\b(a|an|the)\b', ' ', text)

  # Fix extra whitespaces.
  text = ' '.join(text.split())

  return text


def bleu1(targets, predictions):
  """BLEU-1 with normalized targets and predictions.

  Code has been adapted from tweetqa_eval.py since we want this BLEU-1
  calculation to be identical to that used in the TweetQA paper.

  Args:
    targets: list of strings.
    predictions: list of strings.
  Returns:
    A dictionary that looks like: {"bleu": <value>}
  """
  bleu_scores = []
  for target, prediction in zip(targets, predictions):
    target = _normalize_text(target)
    prediction = _normalize_text(prediction)

    # By setting the weights tuple to be (1, 0, 0, 0), only uni-grams are
    # counted towards the BLEU score, resulting in BLEU-1.
    score = sentence_bleu(
        [target.split()], prediction.split(), weights=(1, 0, 0, 0)) * 100

    bleu_scores.append(score)

  return {'bleu1': np.mean(bleu_scores)}


def rouge(targets, predictions):
  """Rouge metrics with normalized targets and predictions.

  Args:
    targets: list of strings.
    predictions: list of strings.
  Returns:
    A dictionary that looks like:
      {"rouge1": <value>, "rouge2": <value>, "rougeLsum": <value>}
  """
  targets = [_normalize_text(x) for x in targets]
  predictions = [_normalize_text(x) for x in predictions]

  return t5_metrics.rouge(targets, predictions)
