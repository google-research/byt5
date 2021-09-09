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

"""Add Tasks to registry."""
import functools
import random

from byt5.tasks import DEFAULT_BYTE_OUTPUT_FEATURES
from byt5.tasks import DEFAULT_MT5_OUTPUT_FEATURES
from byt5.tasks import DEFAULT_PREPROCESSORS
import numpy
import seqio
import t5.data
from t5.data import preprocessors

# Place downloaded data from https://sigmorphon.github.io/sharedtasks/2020 in
# the following directory.
SIGMORPHON_DIR = None

FEATURE_MAP = {
    'byt5': DEFAULT_BYTE_OUTPUT_FEATURES,
    'mt5': DEFAULT_MT5_OUTPUT_FEATURES
}

# ====================== SIGMORPHON-2020 TASK-1 ====================
# Task 1: Multilingual Grapheme-to-Phoneme Conversion
# Please see website https://sigmorphon.github.io/sharedtasks/2020/task1/
# for details.


def get_2020_task1_preprocessor(language):
  return [
      functools.partial(
          preprocessors.preprocess_tsv,
          inputs_format=f' {language} ' + '{0}',
          targets_format='{1}',
          num_fields=2),
  ]


def metrics_task1_2020(targets, predictions):
  """Computes word error rate and edit distance metrics."""

  def edit_distance(x, y) -> int:
    # Implementation from
    # https://github.com/sigmorphon/2020/blob/master/task1/evaluation/evallib.py
    idim = len(x) + 1
    jdim = len(y) + 1
    table = numpy.zeros((idim, jdim), dtype=numpy.uint8)
    table[1:, 0] = 1
    table[0, 1:] = 1
    for i in range(1, idim):
      for j in range(1, jdim):
        if x[i - 1] == y[j - 1]:
          table[i][j] = table[i - 1][j - 1]
        else:
          c1 = table[i - 1][j]
          c2 = table[i][j - 1]
          c3 = table[i - 1][j - 1]
          table[i][j] = min(c1, c2, c3) + 1
    return int(table[-1][-1])

  # Word-level measures.
  correct = 0
  incorrect = 0
  # Label-level measures.
  total_edits = 0
  total_length = 0
  for gold, hypo in zip(targets, predictions):
    edits = edit_distance(gold, hypo)
    length = len(gold)
    if edits == 0:
      correct += 1
    else:
      incorrect += 1
    total_edits += edits
    total_length += length
  wer = incorrect / (correct + incorrect)
  ler = 100 * total_edits / total_length
  return {'wer': wer, 'ler': ler}


langs = [
    'arm', 'bul', 'fre', 'geo', 'hin', 'hun', 'ice', 'kor', 'lit', 'gre', 'ady',
    'dut', 'jpn', 'rum', 'vie'
]
year = '2020'
task = 'task1'
data_dir = f'{SIGMORPHON_DIR}/{year}/{task}/data/'

for lang in langs:
  for prefix, output_features in FEATURE_MAP.items():
    seqio.TaskRegistry.add(
        f'{prefix}_sigmorphon_{year}_{task}.{lang}',
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                'train': f'{data_dir}/train/{lang}_train.tsv',
                'validation': f'{data_dir}/dev/{lang}_dev.tsv',
                'test': f'{data_dir}/test/{lang}_test.tsv'}),
        preprocessors=get_2020_task1_preprocessor(lang) + DEFAULT_PREPROCESSORS,
        output_features=output_features,
        metric_fns=[metrics_task1_2020])

for prefix in ['mt5', 'byt5']:
  t5.data.MixtureRegistry.add(
      f'{prefix}_sigmorphon_{year}_{task}',
      [f'{prefix}_sigmorphon_{year}_{task}.{lang}' for lang in langs],
      default_rate=1.)

# ====================== SIGMORPHON-2020 TASK-0 ====================
# Task 0: Typologically Diverse Morphological Inflection
# Please see website https://sigmorphon.github.io/sharedtasks/2020/task0/
# for details.


def get_2020_task0_preprocessor(language):
  return [
      functools.partial(
          preprocessors.preprocess_tsv,
          inputs_format=f'{language}' + ' {0} ' + 'form={2}',
          targets_format='{1}',
          num_fields=3),
  ]


def metrics_task0_2020(targets, predictions):
  """Calculates exact match and edit distance based metrics."""

  def distance(str1, str2):
    """Levenshtein distance."""
    # Implementation from
    # https://github.com/sigmorphon2020/task0-data/blob/master/evaluate.py
    m = numpy.zeros([len(str2) + 1, len(str1) + 1])
    for x in range(1, len(str2) + 1):
      m[x][0] = m[x - 1][0] + 1
    for y in range(1, len(str1) + 1):
      m[0][y] = m[0][y - 1] + 1
    for x in range(1, len(str2) + 1):
      for y in range(1, len(str1) + 1):
        if str1[y - 1] == str2[x - 1]:
          dg = 0
        else:
          dg = 1
        m[x][y] = min(m[x - 1][y] + 1, m[x][y - 1] + 1, m[x - 1][y - 1] + dg)
    return int(m[len(str2)][len(str1)])

  correct, dist, total = 0., 0., 0.
  for target, prediction in zip(targets, predictions):
    if target == prediction:
      correct += 1
    dist += distance(target, prediction)
    total += 1
  return {
      'accuracy': round(correct / total * 100, 2),
      'distance': round(dist / total, 2)
  }


surprise_lang_path_prefix = [
    'SURPRISE-LANGUAGES/Afro-Asiatic/mlt', 'SURPRISE-LANGUAGES/Germanic/gsw',
    'SURPRISE-LANGUAGES/Nilo-Sahan/dje', 'SURPRISE-LANGUAGES/Romance/frm',
    'SURPRISE-LANGUAGES/Indo-Aryan/urd', 'SURPRISE-LANGUAGES/Uralic/kpv',
    'SURPRISE-LANGUAGES/Sino-Tibetan/bod', 'SURPRISE-LANGUAGES/Germanic/nno',
    'SURPRISE-LANGUAGES/Uralic/olo', 'SURPRISE-LANGUAGES/Romance/fur',
    'SURPRISE-LANGUAGES/Romance/cat', 'SURPRISE-LANGUAGES/Afro-Asiatic/syc',
    'SURPRISE-LANGUAGES/Algic/cre', 'SURPRISE-LANGUAGES/Turkic/kir',
    'SURPRISE-LANGUAGES/Uralic/lud', 'SURPRISE-LANGUAGES/Uralic/udm',
    'SURPRISE-LANGUAGES/Iranian/pus', 'SURPRISE-LANGUAGES/Romance/ast',
    'SURPRISE-LANGUAGES/Germanic/gml', 'SURPRISE-LANGUAGES/Turkic/bak',
    'SURPRISE-LANGUAGES/Indo-Aryan/hin', 'SURPRISE-LANGUAGES/Iranian/fas',
    'SURPRISE-LANGUAGES/Niger-Congo/sna', 'SURPRISE-LANGUAGES/Romance/xno',
    'SURPRISE-LANGUAGES/Romance/vec', 'SURPRISE-LANGUAGES/Dravidian/kan',
    'SURPRISE-LANGUAGES/Afro-Asiatic/orm', 'SURPRISE-LANGUAGES/Turkic/uzb',
    'SURPRISE-LANGUAGES/Uto-Aztecan/ood', 'SURPRISE-LANGUAGES/Turkic/tuk',
    'SURPRISE-LANGUAGES/Iranian/tgk', 'SURPRISE-LANGUAGES/Romance/lld',
    'SURPRISE-LANGUAGES/Turkic/kaz', 'SURPRISE-LANGUAGES/Indo-Aryan/ben',
    'SURPRISE-LANGUAGES/Siouan/dak', 'SURPRISE-LANGUAGES/Romance/glg',
    'SURPRISE-LANGUAGES/Turkic/kjh', 'SURPRISE-LANGUAGES/Turkic/crh',
    'SURPRISE-LANGUAGES/Indo-Aryan/san', 'SURPRISE-LANGUAGES/Dravidian/tel',
    'SURPRISE-LANGUAGES/Tungusic/evn', 'SURPRISE-LANGUAGES/Turkic/aze',
    'SURPRISE-LANGUAGES/Uralic/vro', 'SURPRISE-LANGUAGES/Turkic/uig',
    'SURPRISE-LANGUAGES/Australian/mwf'
]
development_lang_path_prefix = [
    'DEVELOPMENT-LANGUAGES/germanic/swe', 'DEVELOPMENT-LANGUAGES/germanic/ang',
    'DEVELOPMENT-LANGUAGES/oto-manguean/azg',
    'DEVELOPMENT-LANGUAGES/uralic/vep', 'DEVELOPMENT-LANGUAGES/niger-congo/lin',
    'DEVELOPMENT-LANGUAGES/niger-congo/nya',
    'DEVELOPMENT-LANGUAGES/germanic/frr', 'DEVELOPMENT-LANGUAGES/uralic/vot',
    'DEVELOPMENT-LANGUAGES/austronesian/mlg',
    'DEVELOPMENT-LANGUAGES/oto-manguean/ctp',
    'DEVELOPMENT-LANGUAGES/oto-manguean/otm',
    'DEVELOPMENT-LANGUAGES/oto-manguean/ote',
    'DEVELOPMENT-LANGUAGES/uralic/fin',
    'DEVELOPMENT-LANGUAGES/oto-manguean/cpa',
    'DEVELOPMENT-LANGUAGES/austronesian/mao',
    'DEVELOPMENT-LANGUAGES/uralic/mdf', 'DEVELOPMENT-LANGUAGES/germanic/dan',
    'DEVELOPMENT-LANGUAGES/niger-congo/gaa',
    'DEVELOPMENT-LANGUAGES/oto-manguean/cly',
    'DEVELOPMENT-LANGUAGES/uralic/mhr', 'DEVELOPMENT-LANGUAGES/niger-congo/zul',
    'DEVELOPMENT-LANGUAGES/uralic/krl', 'DEVELOPMENT-LANGUAGES/niger-congo/kon',
    'DEVELOPMENT-LANGUAGES/oto-manguean/czn',
    'DEVELOPMENT-LANGUAGES/germanic/gmh', 'DEVELOPMENT-LANGUAGES/uralic/izh',
    'DEVELOPMENT-LANGUAGES/austronesian/ceb',
    'DEVELOPMENT-LANGUAGES/germanic/nob',
    'DEVELOPMENT-LANGUAGES/austronesian/tgl',
    'DEVELOPMENT-LANGUAGES/austronesian/hil',
    'DEVELOPMENT-LANGUAGES/niger-congo/lug',
    'DEVELOPMENT-LANGUAGES/niger-congo/sot',
    'DEVELOPMENT-LANGUAGES/niger-congo/swa',
    'DEVELOPMENT-LANGUAGES/germanic/isl',
    'DEVELOPMENT-LANGUAGES/oto-manguean/pei',
    'DEVELOPMENT-LANGUAGES/uralic/sme', 'DEVELOPMENT-LANGUAGES/germanic/nld',
    'DEVELOPMENT-LANGUAGES/niger-congo/aka',
    'DEVELOPMENT-LANGUAGES/germanic/eng',
    'DEVELOPMENT-LANGUAGES/oto-manguean/zpv',
    'DEVELOPMENT-LANGUAGES/uralic/est', 'DEVELOPMENT-LANGUAGES/uralic/liv',
    'DEVELOPMENT-LANGUAGES/oto-manguean/xty',
    'DEVELOPMENT-LANGUAGES/germanic/deu', 'DEVELOPMENT-LANGUAGES/uralic/myv'
]
year = '2020'
task = 'task0'
data_dir = f'{SIGMORPHON_DIR}/{year}/task0-data/'
langs = [
    path_prefix.split('/')[-1]
    for path_prefix in surprise_lang_path_prefix + development_lang_path_prefix
]
random.shuffle(langs)
path_prefixes = surprise_lang_path_prefix + development_lang_path_prefix

for prefix, output_features in FEATURE_MAP.items():
  for path_prefix in path_prefixes:
    lang = path_prefix.split('/')[-1]
    split_to_filepattern = {
        'train': f'{data_dir}/{path_prefix}.trn',
        'validation': f'{data_dir}/{path_prefix}.dev',
        'test': f'{data_dir}/GOLD-TEST/{lang}.tst',
    }
    seqio.TaskRegistry.add(
        f'{prefix}_sigmorphon_{year}_{task}.{lang}',
        source=seqio.TextLineDataSource(
            split_to_filepattern=split_to_filepattern),
        preprocessors=get_2020_task0_preprocessor(lang) + DEFAULT_PREPROCESSORS,
        output_features=output_features,
        metric_fns=[metrics_task0_2020])

  seqio.TaskRegistry.add(
      f'{prefix}_sigmorphon_{year}_{task}.all',
      source=seqio.TextLineDataSource(
          split_to_filepattern={
              'test': f'{data_dir}/test.tsv',
              'validation': f'{data_dir}/validation.tsv',}),
      preprocessors=[preprocessors.preprocess_tsv,
                     *DEFAULT_PREPROCESSORS,],
      output_features=output_features,
      metric_fns=[metrics_task0_2020])

for prefix in ['mt5', 'byt5']:
  t5.data.MixtureRegistry.add(
      f'{prefix}_sigmorphon_{year}_{task}',
      [f'{prefix}_sigmorphon_{year}_{task}.{lang}' for lang in langs],
      default_rate=1.)
