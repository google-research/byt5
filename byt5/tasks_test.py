# Copyright 2021 The ByT5 Authors.
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

"""Tests for ByT5 tasks."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import byt5.sigmorphon  # pylint:disable=unused-import
import byt5.tasks  # pylint:disable=unused-import
import t5
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.enable_eager_execution()

MixtureRegistry = t5.data.MixtureRegistry
TaskRegistry = t5.data.TaskRegistry
_SEQUENCE_LENGTH = {'inputs': 128, 'targets': 128}

_TASKS = [
    'byt5_dakshina_single_word_translit_indic2latin.bn',
    'byt5_dakshina_word_translit_latin2indic_lang_prefix.bn',
    'byt5_gem_xsum',
    'byt5_mc4.en',
    'byt5_sigmorphon_2020_task1.ar',
    'byt5_super_glue_boolq_v102',
    'byt5_super_glue_cb_v102',
    'byt5_super_glue_copa_v102',
    'byt5_super_glue_multirc_v102',
    'byt5_super_glue_record_v102',
    'byt5_super_glue_rte_v102',
    'byt5_super_glue_wic_v102',
    'byt5_super_glue_wsc_v102_simple_eval',
    'byt5_super_glue_wsc_v102_simple_train',
    'byt5_tweetqa',
    'byt5_wiki.en',
    'byt5_wmt15_enfr_v003',
    'byt5_wmt16_enro_v003',
    'byt5_wmt_t2t_ende_v003',
    'char_t5_mc4.en',
    'mt5_dakshina_single_word_translit_indic2latin.bn',
    'mt5_dakshina_word_translit_latin2indic_lang_prefix.bn',
    'mt5_sigmorphon_2020_task0.dje'
]

_MIXTURES = [
    'byt5_dak_wrdtrnslit_ind2lat',
    'byt5_dak_wrdtrnslit_lat2ind',
    'byt5_dak_wrdtrnslit_lat2ind_lp',
    'byt5_glue_v002_proportional',
    'byt5_mlqa_translate_train',
    'byt5_mlqa_zeroshot',
    'byt5_ner_multilingual',
    'byt5_ner_zeroshot',
    'byt5_pawsx_translate_train',
    'byt5_pawsx_zeroshot',
    'byt5_sigmorphon_2020_task0',
    'byt5_sigmorphon_2020_task1',
    'byt5_super_glue_v102_proportional',
    'byt5_tydiqa',
    'byt5_tydiqa_translate_train',
    'byt5_tydiqa_zeroshot',
    'byt5_wikilingua',
    'byt5_xnli_translate_train',
    'byt5_xnli_zeroshot',
    'byt5_xquad_translate_train',
    'byt5_xquad_zeroshot',
    'mt5_dak_wrdtrnslit_ind2lat',
    'mt5_dak_wrdtrnslit_lat2ind',
    'mt5_dak_wrdtrnslit_lat2ind_lp',
    'mt5_sigmorphon_2020_task0',
    'mt5_sigmorphon_2020_task1'
]


class TasksTest(parameterized.TestCase):

  @parameterized.parameters(((name,) for name in _TASKS))
  def test_task(self, name):
    task = TaskRegistry.get(name)
    split = 'train' if 'train' in task.splits else 'validation'
    logging.info('task=%s, split=%s', name, split)
    ds = task.get_dataset(_SEQUENCE_LENGTH, split)
    for d in ds:
      logging.info(d)
      break

  @parameterized.parameters(((name,) for name in _MIXTURES))
  def test_mixture(self, name):
    mixture = MixtureRegistry.get(name)
    logging.info('mixture=%s', name)
    ds = mixture.get_dataset(_SEQUENCE_LENGTH, 'train')
    for d in ds:
      logging.info(d)
      break


if __name__ == '__main__':
  absltest.main()
