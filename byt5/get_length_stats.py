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

r"""Get the stats of input and target length for T5/mT5/ByT5 tasks.

If the input length is too long which causes OOM, one could truncate the length
to 1.1 * 99th percentile length.

"""

from absl import app
from absl import flags
import byt5.tasks  # pylint:disable=unused-import
import multilingual_t5.tasks  # pylint:disable=unused-import
import t5



FLAGS = flags.FLAGS
flags.DEFINE_string("task_or_mixture", None, "task/mixture name")
flags.DEFINE_string("split", "train", "train, validation, or test.")
flags.DEFINE_boolean("use_cached", True, "Use cached or not.")
flags.DEFINE_integer("num_samples", None,
                     "Number of samples to take to compute the lengths.")


MAX_LENGH = int(10e8)


def get_perc_99_len(input_length):
  """Get 99 percentile sequence length."""
  lengths = sorted(input_length)
  perc_99 = len(input_length) * 99 // 100
  perc_99_len = lengths[perc_99]
  return perc_99_len


def get_stats(task_or_mixture, split="train"):
  """Get task length stats.

  Args:
    task_or_mixture: string, task or mixture name.
    split: string, split.
  """
  print(f"Get length statistics for {task_or_mixture} {split}...")
  if task_or_mixture in t5.data.seqio.TaskRegistry.names():
    data = t5.data.seqio.TaskRegistry.get(task_or_mixture)
  elif task_or_mixture in t5.data.seqio.MixtureRegistry.names():
    data = t5.data.seqio.MixtureRegistry.get(task_or_mixture)
  else:
    raise ValueError(f"{task_or_mixture} {split} is not registered.")

  data = data.get_dataset(split=split,
                          sequence_length={"inputs": MAX_LENGH,
                                           "targets": MAX_LENGH},
                          use_cached=FLAGS.use_cached,
                          num_epochs=1)

  if FLAGS.num_samples: data = data.take(FLAGS.num_samples)
  ds = data.as_numpy_iterator()

  input_lengths, target_lengths = [], []
  for ex in ds:
    input_lengths.append(len(ex["inputs"]))
    target_lengths.append(len(ex["targets"]))
  recommend_input_length = 1.1 * get_perc_99_len(input_lengths)

  output = (f"Total # of examples in {split}: {len(input_lengths)}\n"
            f"Min input length: {min(input_lengths)}\n"
            f"Max input length: {max(input_lengths)}\n"
            f"1.1 * 99 percertile of length: {recommend_input_length}\n"
            f"Min target length: {min(target_lengths)}\n"
            f"Max target length: {max(target_lengths)}\n")

  print(output)


def main(_):
  get_stats(FLAGS.task_or_mixture, split=FLAGS.split)

if __name__ == "__main__":
  app.run(main)
