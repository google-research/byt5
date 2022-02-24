# Copyright 2022 The ByT5 Authors.
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

"""Tests for byt5.metrics."""

from absl.testing import absltest
from byt5 import metrics


class MetricsTest(absltest.TestCase):

  def test_edit_distance(self):

    d = metrics.cer(["abcd", "aaaa", "ab"],
                    ["abde", "bbbb", "a"])

    # abcd -> abde (2 edits). Target length = 4.
    # aaaa -> bbbb (4 edits). Target length = 4.
    # ab -> a (1 edit). Target length = 2.
    # CER = Total # of edits / Total # of target chars = 7 / 10 = 0.7
    self.assertDictEqual(d, {"cer": 0.7})

  def test_normalize_text(self):
    output = metrics._normalize_text(" THIS is a   string!")
    expected = "this is string"
    self.assertEqual(output, expected)

  def test_bleu1_full_match(self):
    targets = ["this is a string", "this is a string", "this is a string",
               "this is a string"]
    predictions = ["this is a string", "THIS is a string", "this is a string!",
                   "this is string"]
    d = metrics.bleu1(targets, predictions)

    # Normalization should remove articles, extra spaces and punctuations,
    # resulting in a BLEU-1 score of 100.0.
    self.assertDictEqual(d, {"bleu1": 100.0})

  def test_bleu1_no_full_match(self):
    targets = ["this is a string"]
    predictions = ["this is not a string"]
    d = metrics.bleu1(targets, predictions)

    self.assertDictEqual(d, {"bleu1": 75.0})

  def test_rouge_full_match(self):
    targets = ["this is a string", "this is a string", "this is a string",
               "this is a string"]
    predictions = ["this is a string", "THIS is a string", "this is a string!",
                   "this is string"]

    d = metrics.rouge(targets, predictions)
    self.assertDictEqual(d, {"rouge1": 100, "rouge2": 100, "rougeLsum": 100})

  def test_rouge_no_match(self):
    targets = ["this is a string"]
    predictions = ["", ""]
    d = metrics.rouge(targets, predictions)

    self.assertDictEqual(d, {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0})


if __name__ == "__main__":
  absltest.main()
