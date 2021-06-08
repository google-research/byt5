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

"""Add ByT5 Tasks to registry."""
import functools

import byt5.metrics as byt5_metrics
from multilingual_t5 import preprocessors
from multilingual_t5 import utils
from multilingual_t5.evaluation import metrics as mt5_metrics
from multilingual_t5.tasks import DEFAULT_OUTPUT_FEATURES as DEFAULT_MT5_OUTPUT_FEATURES

import t5.data
import t5.data.tasks
from t5.evaluation import metrics
import tensorflow_datasets as tfds



MEAN_NOISE_SPAN_LENGTH = 20
DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

DEFAULT_BYTE_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(vocabulary=t5.data.ByteVocabulary()),
    "targets": t5.data.Feature(vocabulary=t5.data.ByteVocabulary())
}

FEATURE_MAP = {
    "byt5": DEFAULT_BYTE_OUTPUT_FEATURES,
    "mt5": DEFAULT_MT5_OUTPUT_FEATURES
}

MC4_LANGS = tfds.text.c4.MC4_LANGUAGES

# Multilingual BERT was trained on 104 languages. We include 103 of these
# languages, as tfds.wikipedia doesn't distinguish between simplified and
# traditional Chinese, and only contains "zh" (which is a mix of simplified
# and traditional).
# https://github.com/google-research/bert/blob/master/multilingual.md
WIKI_LANGS = [
    "af", "an", "ar", "ast", "az", "azb", "ba", "bar", "be", "bg", "bn", "bpy",
    "br", "bs", "ca", "ce", "ceb", "cs", "cv", "cy", "da", "de", "el", "en",
    "es", "et", "eu", "fa", "fi", "fr", "fy", "ga", "gl", "gu", "he", "hi",
    "hr", "ht", "hu", "hy", "id", "io", "is", "it", "ja", "jv", "ka", "kk",
    "kn", "ko", "ky", "la", "lb", "lmo", "lt", "lv", "mg", "min", "mk", "ml",
    "mn", "mr", "ms", "my", "nds-nl", "ne", "new", "nl", "nn", "no", "oc",
    "pa", "pl", "pms", "pnb", "pt", "ro", "ru", "scn", "sco", "sh", "sk", "sl",
    "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tl", "tr", "tt",
    "uk", "ur", "uz", "vi", "vo", "war", "yo", "zh"
]

# =========================== Pretraining Tasks/Mixtures =======================
# mC4
for lang in MC4_LANGS:
  t5.data.TaskRegistry.add(
      "byt5_mc4.{}".format(lang.replace("-", "_")),
      t5.data.TfdsTask,
      tfds_name="c4/multilingual:3.0.1",
      splits={"train": lang,
              "validation": f"{lang}-validation"},
      text_preprocessor=functools.partial(
          t5.data.preprocessors.rekey,
          key_map={"inputs": None, "targets": "text"}),
      token_preprocessor=functools.partial(
          t5.data.preprocessors.span_corruption,
          mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH),
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[])

mc4 = (["byt5_mc4.{}".format(lang.replace("-", "_"))
        for lang in MC4_LANGS])
t5.data.MixtureRegistry.add(
    "byt5_mc4", mc4, default_rate=DEFAULT_MIX_RATE)

# Wikipedia
for lang in WIKI_LANGS:
  t5.data.TaskRegistry.add(
      "byt5_wiki.{}".format(lang.replace("-", "_")),
      t5.data.TfdsTask,
      tfds_name="wikipedia/20200301.{}:1.0.0".format(lang),
      text_preprocessor=[
          functools.partial(
              t5.data.preprocessors.rekey,
              key_map={
                  "inputs": None,
                  "targets": "text"
              }),
      ],
      token_preprocessor=functools.partial(
          t5.data.preprocessors.span_corruption,
          mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH),
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[])

wiki = ["byt5_wiki.{}".format(lang.replace("-", "_")) for lang in WIKI_LANGS]
t5.data.MixtureRegistry.add("byt5_wiki", wiki, default_rate=DEFAULT_MIX_RATE)

# Mixture of mC4 and WIKI
t5.data.MixtureRegistry.add(
    "byt5_mc4_wiki", mc4 + wiki, default_rate=DEFAULT_MIX_RATE)

# =========================== Fine-tuning Tasks/Mixtures =======================
# ----- XNLI -----
# XNLI zero-shot task. This fine-tunes on English MNLI training data and then
# evaluates on multilingual XNLI dev/test data.

XNLI_LANGS = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
    "ur", "vi", "zh"
]

t5.data.TaskRegistry.add(
    "byt5_xnli_train",
    t5.data.TfdsTask,
    tfds_name="multi_nli:1.1.0",
    splits=["train"],
    text_preprocessor=preprocessors.process_mnli,
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    metric_fns=[metrics.accuracy])
for lang in XNLI_LANGS:
  t5.data.TaskRegistry.add(
      "byt5_xnli_dev_test.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="xnli:1.1.0",
      splits=["validation", "test"],
      text_preprocessor=[
          functools.partial(
              preprocessors.process_xnli, target_languages=[lang])
      ],
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[metrics.accuracy])
  if lang == "en":
    continue
  t5.data.TaskRegistry.add(
      "byt5_xnli_translate_train.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="xtreme_xnli:1.1.0",
      splits=["train"],
      text_preprocessor=[
          functools.partial(
              preprocessors.process_xnli, target_languages=[lang])
      ],
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[metrics.accuracy])
t5.data.TaskRegistry.add(
    "byt5_xnli_dev_test.all_langs",
    t5.data.TfdsTask,
    tfds_name="xnli:1.1.0",
    splits=["validation", "test"],
    text_preprocessor=[
        functools.partial(
            preprocessors.process_xnli, target_languages=XNLI_LANGS)
    ],
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    metric_fns=[metrics.accuracy])
xnli_zeroshot = (["byt5_xnli_train", "byt5_xnli_dev_test.all_langs"] +
                 ["byt5_xnli_dev_test.{}".format(lang) for lang in XNLI_LANGS])
t5.data.MixtureRegistry.add("byt5_xnli_zeroshot",
                            xnli_zeroshot,
                            default_rate=1.0)
xnli_translate_train = xnli_zeroshot + [
    "byt5_xnli_translate_train.{}".format(lang)
    for lang in XNLI_LANGS
    if lang != "en"
]
t5.data.MixtureRegistry.add(
    "byt5_xnli_translate_train", xnli_translate_train, default_rate=1.0)

# ----- PAWS -----
label_names = ["different_meaning", "paraphrase"]
text_preprocessor = [
    functools.partial(
        t5.data.preprocessors.glue,
        benchmark_name="paws",
        label_names=label_names,
        feature_names=["sentence1", "sentence2"],
        id_key=None)
]

postprocess_fn = functools.partial(
        t5.data.postprocessors.string_label_to_class_id,
        label_classes=label_names)

t5.data.TaskRegistry.add(
    "byt5_paws",
    t5.data.TfdsTask,
    tfds_name="paws_x_wiki/en:1.0.0",
    splits=["train"],
    text_preprocessor=text_preprocessor,
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    postprocess_fn=postprocess_fn,
    metric_fns=[metrics.accuracy])

for lang in utils.PAWSX_LANGS:
  t5.data.TaskRegistry.add(
      "byt5_pawsx_dev_test.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="paws_x_wiki/{}:1.0.0".format(lang),
      splits=["validation", "test"],
      text_preprocessor=text_preprocessor,
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      postprocess_fn=postprocess_fn,
      metric_fns=[metrics.accuracy])

  # This uses machine translations provided by the PAWS-X paper.
  t5.data.TaskRegistry.add(
      "byt5_pawsx_translate_train_original.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="paws_x_wiki/{}:1.0.0".format(lang),
      splits=["train"],
      text_preprocessor=text_preprocessor,
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      postprocess_fn=postprocess_fn,
      metric_fns=[metrics.accuracy])

  if lang != "en":
  # This uses machine translations provided by the XTREME paper.
    t5.data.TaskRegistry.add(
        "byt5_pawsx_translate_train.{}".format(lang),
        t5.data.TfdsTask,
        tfds_name="xtreme_pawsx/{}:1.0.0".format(lang),
        splits=["train"],
        text_preprocessor=text_preprocessor,
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        postprocess_fn=postprocess_fn,
        metric_fns=[metrics.accuracy])

t5.data.TaskRegistry.add(
    "byt5_pawsx_dev_test.all_langs",
    t5.data.Task,
    splits=["validation", "test"],
    dataset_fn=utils.pawsx_all_langs_dataset_fn,
    text_preprocessor=text_preprocessor,
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    postprocess_fn=postprocess_fn,
    metric_fns=[metrics.accuracy])

# PAWSX Zero-Shot
pawsx_eval = [
    "byt5_pawsx_dev_test.{}".format(lang) for lang in utils.PAWSX_LANGS
] + ["byt5_pawsx_dev_test.all_langs"]
pawsx = ["byt5_paws"] +  pawsx_eval
t5.data.MixtureRegistry.add("byt5_pawsx_zeroshot", pawsx, default_rate=1.0)

pawsx_translate_train = ["byt5_paws"] + [
    "byt5_pawsx_translate_train.{}".format(lang)
    for lang in utils.PAWSX_LANGS
    if lang != "en"
] + pawsx_eval
t5.data.MixtureRegistry.add(
    "byt5_pawsx_translate_train", pawsx_translate_train, default_rate=1.0)

pawsx_translate_train_original = [
    "byt5_pawsx_translate_train_original.{}".format(lang)
    for lang in utils.PAWSX_LANGS
] + pawsx_eval
t5.data.MixtureRegistry.add(
    "byt5_pawsx_translate_train_original",
    pawsx_translate_train,
    default_rate=1.0)


# ----- TyDiQA GoldP-----
# The "validation" split contains all the validation examples for all the
# individual languages together.
TYDIQA_LANGS = ["ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te"]
t5.data.TaskRegistry.add(
    "byt5_tydiqa_train_dev",
    t5.data.TfdsTask,
    tfds_name="tydi_qa/goldp:2.1.0",
    splits=["train", "validation"],
    text_preprocessor=preprocessors.xquad,
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    metric_fns=[metrics.squad])

for lang in TYDIQA_LANGS:
  t5.data.TaskRegistry.add(
      "byt5_tydiqa_dev.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="tydi_qa/goldp:2.1.0",
      splits={"validation": "validation-{}".format(lang)},
      text_preprocessor=preprocessors.xquad,
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[metrics.squad])

tydiqa = (["byt5_tydiqa_train_dev"] +
          ["byt5_tydiqa_dev.{}".format(lang) for lang in TYDIQA_LANGS])
t5.data.MixtureRegistry.add("byt5_tydiqa", tydiqa, default_rate=1.0)

# ----- TyDiQA GoldP Zero-Shot-----
# This Zero-Shot setting matches the XTREME setup, where training is done on
# the English data of TyDiQa. In the TyDiQA paper, fine-tuning was done on
# SQuAD for zero-shot evaluation.
TYDIQA_LANGS = ["ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te"]
t5.data.TaskRegistry.add(
    "byt5_tydiqa_train.en",
    t5.data.TfdsTask,
    tfds_name="tydi_qa/goldp:2.1.0",
    splits=["train"],
    text_preprocessor=[
        preprocessors.xquad,
        functools.partial(
            preprocessors.filter_tydiqa_by_language, lang="english")
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    metric_fns=[metrics.squad])


tydiqa_zeroshot = (["byt5_tydiqa_train.en"] +
                   ["byt5_tydiqa_dev.{}".format(lang) for lang in TYDIQA_LANGS])
t5.data.MixtureRegistry.add(
    "byt5_tydiqa_zeroshot", tydiqa_zeroshot, default_rate=1.0)


# Defining translate-train tasks.
for lang in TYDIQA_LANGS:
  # Skipping English, since translate-train is not available.
  if lang == "en":
    continue
  t5.data.TaskRegistry.add(
      "byt5_tydiqa_translate_train.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="tydi_qa/goldp:2.1.0",
      splits={"train": "translate-train-{}".format(lang)},
      text_preprocessor=preprocessors.xquad,
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[metrics.squad])

tydiqa_translate_train = (
    ["byt5_tydiqa_train.en"]
    + [f"byt5_tydiqa_translate_train.{lang}"
       for lang in TYDIQA_LANGS if lang != "en"]
    + [f"byt5_tydiqa_dev.{lang}" for lang in TYDIQA_LANGS])
t5.data.MixtureRegistry.add(
    "byt5_tydiqa_translate_train", tydiqa_translate_train, default_rate=1.0)

# ----- English SQUAD -----
t5.data.TaskRegistry.add(
    "byt5_squad_train_dev",
    t5.data.TfdsTask,
    tfds_name="squad/v1.1:3.0.0",
    splits=["train", "validation"],
    text_preprocessor=preprocessors.xquad,
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    metric_fns=[metrics.squad])

# ----- XQuAD -----
for lang in utils.XQUAD_LANGS_TRAIN_DEV:
  t5.data.TaskRegistry.add(
      "byt5_xquad_translate_train_dev.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="xquad/{}:3.0.0".format(lang),
      splits={
          "train": "translate-train",
          "validation": "translate-dev"
      },
      text_preprocessor=preprocessors.xquad,
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[metrics.squad])

for lang in utils.XQUAD_LANGS_TEST:
  t5.data.TaskRegistry.add(
      "byt5_xquad_test.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="xquad/{}:3.0.0".format(lang),
      splits=["test"],
      text_preprocessor=preprocessors.xquad,
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[metrics.squad])

# Additional test task containing all the languages.
t5.data.TaskRegistry.add(
    "byt5_xquad_test.all_langs",
    t5.data.Task,
    splits=["test"],
    dataset_fn=utils.xquad_all_langs_dataset_fn,
    text_preprocessor=preprocessors.xquad,
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    metric_fns=[metrics.squad])

# XQuAD Zero-Shot (SQuAD train, SQuAD dev, XQuAD test).
xquad_test = (["byt5_xquad_test.{}".format(lang)
               for lang in utils.XQUAD_LANGS_TEST])
xquad_zeroshot = (["byt5_squad_train_dev", "byt5_xquad_test.all_langs"] +
                  xquad_test)
t5.data.MixtureRegistry.add("byt5_xquad_zeroshot",
                            xquad_zeroshot,
                            default_rate=1.0)

# XQuAD Translate-Train (English SQuAD, XQuAD translate-train,
# XQuAD translate-dev, XQuAD test)
# Note that the QA translate-train baselines from Hu et al (XTREME)
# do not include the English data. However, Fang et al (FILTER) do include
# English data.
xquad_translate_train = [
    "byt5_xquad_translate_train_dev.{}".format(lang)
    for lang in utils.XQUAD_LANGS_TRAIN_DEV
] + ["byt5_squad_train_dev"] +  ["byt5_xquad_test.all_langs"] + xquad_test
t5.data.MixtureRegistry.add(
    "byt5_xquad_translate_train", xquad_translate_train, default_rate=1.0)

# ----- MLQA -----
MLQA_LANGS = ["ar", "de", "en", "es", "hi", "vi", "zh"]

for lang in MLQA_LANGS:
  t5.data.TaskRegistry.add(
      "byt5_mlqa_dev_test.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="mlqa/{}:1.0.0".format(lang),
      splits=["validation", "test"],
      text_preprocessor=preprocessors.xquad,
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[functools.partial(mt5_metrics.mlqa, lang=lang)])

# MLQA Zero-Shot
mlqa_dev_test = [f"byt5_mlqa_dev_test.{lang}" for lang in MLQA_LANGS]
mlqa_zeroshot = ["byt5_squad_train_dev"] + mlqa_dev_test
t5.data.MixtureRegistry.add("byt5_mlqa_zeroshot",
                            mlqa_zeroshot,
                            default_rate=1.0)

# MLQA Translate-Train
mlqa_translate_train = [
    "byt5_xquad_translate_train_dev.{}".format(lang)
    for lang in MLQA_LANGS
    if lang != "en"
] + ["byt5_squad_train_dev"] + mlqa_dev_test

t5.data.MixtureRegistry.add(
    "byt5_mlqa_translate_train", mlqa_translate_train, default_rate=1.0)


# Wikilingua
# The version of Wikilingua here is the one from the GEM dataset:
# https://arxiv.org/pdf/2102.01672.pdf.
# Note: It was found that there is some data leakage in the GEM WikiLingua
# data with English target summaries in the test set of one language
# appearing in the training set of another language. Training in a multi-task
# setup is not recommended till this issue is resolved.
# TODO(adityabarua): Remove this comment when the issue is resolved.
WIKILINGUA_LANGS = ["es", "ru", "tr", "vi"]

for model, features in FEATURE_MAP.items():
  for lang in WIKILINGUA_LANGS:
    t5.data.TaskRegistry.add(
        "{model}_wikilingua.{lang}".format(model=model, lang=lang),
        t5.data.TfdsTask,
        tfds_name="gem/wiki_lingua_{lang}_en:1.0.1".format(lang=lang),
        text_preprocessor=[
            functools.partial(
                t5.data.preprocessors.rekey,
                key_map={
                    "inputs": "source",
                    "targets": "target"
                }),
        ],
        output_features=features,
        metric_fns=[metrics.rouge])

  t5.data.MixtureRegistry.add(
      "{model}_wikilingua".format(model=model),
      ["{model}_wikilingua.{lang}".format(model=model, lang=lang) for
       lang in WIKILINGUA_LANGS],
      default_rate=1.0)

# SuperGLUE
for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
  # We use a simplified version of WSC, defined below
  if "wsc" in b.name:
    continue
  if b.name == "axb":
    text_preprocessor = [
        functools.partial(
            t5.data.preprocessors.rekey,
            key_map={
                "premise": "sentence1",
                "hypothesis": "sentence2",
                "label": "label",
                "idx": "idx",
            }),
        t5.data.glue_utils.get_glue_text_preprocessor(b),
    ]
  else:
    text_preprocessor = t5.data.glue_utils.get_glue_text_preprocessor(b)

  t5.data.TaskRegistry.add(
      "byt5_super_glue_%s_v102" % b.name,
      t5.data.TfdsTask,
      tfds_name="super_glue/%s:1.0.2" % b.name,
      text_preprocessor=text_preprocessor,
      metric_fns=t5.data.glue_utils.get_super_glue_metric(b.name),
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      postprocess_fn=t5.data.glue_utils.get_glue_postprocess_fn(b),
      splits=["test"] if b.name in ["axb", "axg"] else None)

# Definitely Pronoun Resolution
t5.data.TaskRegistry.add(
    "byt5_dpr_v001_simple",
    t5.data.TfdsTask,
    tfds_name="definite_pronoun_resolution:1.1.0",
    text_preprocessor=t5.data.preprocessors.definite_pronoun_resolution_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES)

# WSC
t5.data.TaskRegistry.add(
    "byt5_super_glue_wsc_v102_simple_train",
    t5.data.TfdsTask,
    tfds_name="super_glue/wsc.fixed:1.0.2",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.wsc_simple, correct_referent_only=True),
    metric_fns=[],
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    splits=["train"])
t5.data.TaskRegistry.add(
    "byt5_super_glue_wsc_v102_simple_eval",
    t5.data.TfdsTask,
    tfds_name="super_glue/wsc.fixed:1.0.2",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.wsc_simple, correct_referent_only=False),
    postprocess_fn=t5.data.postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    splits=["validation", "test"])

_byt5_super_glue_tasks = {}
for task, value in t5.data.get_super_glue_weight_mapping().items():
  _byt5_super_glue_tasks["byt5_" + task] = value

t5.data.MixtureRegistry.add(
    "byt5_super_glue_v102_proportional",
    list(_byt5_super_glue_tasks.items()))


# GLUE
for b in tfds.text.glue.Glue.builder_configs.values():
  t5.data.TaskRegistry.add(
      "byt5_glue_%s_v002" % b.name,
      t5.data.TfdsTask,
      tfds_name="glue/%s:1.0.0" % b.name,
      text_preprocessor=t5.data.get_glue_text_preprocessor(b),
      metric_fns=t5.data.get_glue_metric(b.name),
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      postprocess_fn=t5.data.get_glue_postprocess_fn(b),
      splits=["test"] if b.name == "ax" else None,
  )

t5.data.MixtureRegistry.add(
    "byt5_glue_v002_proportional",
    list({
        f"byt5_{k}": v
        for k, v in t5.data.glue_utils.get_glue_weight_mapping().items()
    }.items()))

# ------ WMT ----------
for prefix, b, tfds_version in t5.data.tasks.b_configs:
  t5.data.TaskRegistry.add(
      "byt5_wmt%s_%s%s_v003" % (prefix, b.language_pair[1], b.language_pair[0]),
      t5.data.TfdsTask,
      tfds_name="wmt%s_translate/%s:%s" % (prefix, b.name, tfds_version),
      text_preprocessor=functools.partial(
          t5.data.preprocessors.translate,
          source_language=b.language_pair[1],
          target_language=b.language_pair[0],
          ),
      metric_fns=[metrics.bleu],
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES)

# Special case for t2t ende.
b = tfds.translate.wmt_t2t.WmtT2tTranslate.builder_configs["de-en"]
t5.data.TaskRegistry.add(
    "byt5_wmt_t2t_ende_v003",
    t5.data.TfdsTask,
    tfds_name="wmt_t2t_translate/de-en:1.0.0",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.translate,
        source_language=b.language_pair[1],
        target_language=b.language_pair[0],
        ),
    metric_fns=[metrics.bleu],
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES)

# ----- WikiAnn NER -----

NER_LANGS = [
    "af", "ar", "bg", "bn", "de", "el", "en", "es", "et", "eu", "fa", "fi",
    "fr", "he", "hi", "hu", "id", "it", "ja", "jv", "ka", "kk", "ko", "ml",
    "mr", "ms", "my", "nl", "pt", "ru", "sw", "ta", "te", "th", "tl", "tr",
    "ur", "vi", "yo", "zh"
]

for lang in NER_LANGS:
  t5.data.TaskRegistry.add(
      "byt5_ner_train.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="wikiann/{}:1.0.0".format(lang),
      splits=["train"],
      text_preprocessor=preprocessors.wikiann,
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[mt5_metrics.span_f1])

  t5.data.TaskRegistry.add(
      "byt5_ner_eval.{}".format(lang),
      t5.data.TfdsTask,
      tfds_name="wikiann/{}:1.0.0".format(lang),
      splits=["validation", "test"],
      text_preprocessor=preprocessors.wikiann,
      output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
      metric_fns=[mt5_metrics.span_f1])

# NER zero-shot
t5.data.MixtureRegistry.add(
    "byt5_ner_zeroshot", ["byt5_ner_train.{}".format("en")] +
    ["byt5_ner_eval.{}".format(lang) for lang in NER_LANGS],
    default_rate=1.0)

# NER multilingual
t5.data.MixtureRegistry.add(
    "byt5_ner_multilingual",
    ["byt5_ner_train.{}".format(lang) for lang in NER_LANGS] +
    ["byt5_ner_eval.{}".format(lang) for lang in NER_LANGS],
    default_rate=1.0)


