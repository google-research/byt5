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

r"""Given a task and its input length, return the max target length with packing.

Use a target length much longer than you'd expect given the number of packed
examples for the specified input length. Model will not learn well if target
lengths are too small.

"""

from absl import app
from absl import flags
from byt5 import tasks as byt5_tasks
from multilingual_t5 import tasks as mt5_tasks
import numpy as np
import t5.models.mesh_transformer as mesh_transformer
import tensorflow_datasets as tfds



FLAGS = flags.FLAGS
flags.DEFINE_string("task_or_mixture", None, "task/mixture name")
flags.DEFINE_integer("input_length", None, "Input length of tasks.")
flags.DEFINE_integer("num_sample_examples", 100,
                     "Set to None to load all examples.")
flags.DEFINE_string("vocab", None, "mt5 or byt5")
flags.DEFINE_list("splits", "train,dev,test", "Comma-separated list of splits.")


# Vocabs
VOCABS = {"byt5": byt5_tasks.DEFAULT_BYTE_OUTPUT_FEATURES,
          "mt5": mt5_tasks.DEFAULT_OUTPUT_FEATURES}

# Splits
SPLITS = {"train": tfds.Split.TRAIN,
          "dev": tfds.Split.VALIDATION,
          "test": tfds.Split.TEST}


def compute_target_length(task_or_mixture,
                          input_len,
                          num_sample_examples,
                          vocab):
  """Given a task and a input length, write the max target lengths with packing.

  Args:
    task_or_mixture: string, task or mixture name.
    input_len: int, input length.
    num_sample_examples: int, number of sample examples.
    vocab: string, vocab.
  """
  target_lengths = []
  for split in FLAGS.splits:
    dummy_target_len = 512
    ds = mesh_transformer.mesh_train_dataset_fn(
        task_or_mixture,
        sequence_length={"inputs": input_len, "targets": dummy_target_len},
        vocabulary=vocab,
        dataset_split=SPLITS[split])
    if num_sample_examples:
      ds = ds.take(num_sample_examples)
    lengths = ([np.argmin(ex["targets_segmentation"])
                for ex in ds.as_numpy_iterator()])
    target_lengths.append(np.max(lengths))

  # Print results
  splits = FLAGS.splits + ["all"]
  target_lengths = target_lengths + [np.max(target_lengths)]
  for split, max_len in zip(splits, target_lengths):
    print(f"Max target length for {split} is: {max_len} .")



def main(_):
  if FLAGS.vocab not in VOCABS:
    raise ValueError(f"Vocab of {FLAGS.vcoab} is not available")
  compute_target_length(FLAGS.task_or_mixture,
                        FLAGS.input_length,
                        FLAGS.num_sample_examples,
                        FLAGS.vocab)


if __name__ == "__main__":
  app.run(main)
