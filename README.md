# ByT5: Towards a token-free future with pre-trained byte-to-byte models

ByT5 is a tokenizer-free extension of the [mT5
model](https://arxiv.org/abs/2010.11934). Instead of using a subword vocabulary
like most other pretrained language models (BERT, XLM-R, T5, GPT-3), our ByT5
model operates directly on UTF-8 bytes, removing the need for any text
preprocessing. Beyond the reduction in system complexity, we find that
parameter-matched ByT5 models are competitive with mT5 across a range of tasks,
and outperform mT5 on tasks that involve noisy text or are sensitive to
spelling and pronunciation. This repo can be used to reproduce the experiments
in the [ByT5 paper][paper].

## Usage

### Training

To run this code, you need to install the [t5
library](https://pypi.org/project/t5/). General instructions for training,
fine-tuning, evaluation, and exporting models for inference can be found in the
[t5
repo](https://github.com/google-research/text-to-text-transfer-transformer). In
order to use the additional ByT5 tasks provided in this library with the
`t5_mesh_transformer` command, run from this directory and add the flag
`--module_import="byt5.tasks"`.

To train a `ByT5-Large` model on the
[mc4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual_nights_stay)
task from scratch as described in the paper:

```
export PROJECT=yourproject
export ZONE=yourzone
export BUCKET=yourbucket
export TPU=yourtpu

ctpu up --name=$TPU --project=$PROJECT --zone=$ZONE --tpu-size=v3-256 --tpu-only --noconf

TASK=byt5_mc4
MODEL_DIR="${BUCKET}${TASK}"

python -m t5.models.mesh_transformer_main \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="models/byt5.large.gin" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.sequence_length = {'inputs': 1024, 'targets': 189}" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 1048576)" \
  --gin_param="mean_noise_span_length = 20" \
  --gin_param="utils.run.learning_rate_schedule=@learning_rate_schedules.rsqrt_no_ramp_down" \
  --gin_param="run.train_steps = 1000000" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-256'" \
  --eval_mode="perplexity_eval" \
  --eval_gin_param="mesh_eval_dataset_fn.num_eval_examples = 10000" \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
  --module_import="byt5.tasks"
```

### Fine-Tuning

The example below shows how to finetune the `ByT5-Large` model on the XNLI
zeroshot task.

```
export PROJECT=yourproject
export ZONE=yourzone
export BUCKET=yourbucket
export TPU=yourtpu

ctpu up --name=$TPU --project=$PROJECT --zone=$ZONE --tpu-size=v3-256 --tpu-only --noconf

TASK=byt5_xnli_zeroshot
PRETRAINED_DIR=gs://t5-data/pretrained_models/byt5/large
PRETRAINED_STEPS=1000000
FINETUNE_STEPS=262144
MODEL_DIR="${BUCKET}${TASK}"

# Run fine-tuning
python -m t5.models.mesh_transformer_main \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-256'" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS+FINETUNE_STEPS))" \
  --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
  --module_import="byt5.tasks"
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 1048576)" \
  --gin_param="utils.run.sequence_length = {'inputs': 2048, 'targets': 56}"
  --eval_gin_param="Bitransformer.decode.max_decode_length = 56" \
```

The remaining experiments are shown in the [tasks.py](byt5/tasks.py) file.

## Released Model Checkpoints

We have released the following checkpoints for pre-trained models described in
our [paper][paper]:

* **ByT5-Small** (300 million parameters): [gs://t5-data/pretrained_models/byt5/small](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/byt5/small/)
* **ByT5-Base** (580 million parameters): [gs://t5-data/pretrained_models/byt5/base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/byt5/base/)
* **ByT5-Large** (1.2 billion parameters): [gs://t5-data/pretrained_models/byt5/large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/byt5/large/)
* **ByT5-XL** (3.7 billion parameters): [gs://t5-data/pretrained_models/byt5/xl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/byt5/xl/)
* **ByT5-XXL** (13 billion parameters): [gs://t5-data/pretrained_models/byt5/xxl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/byt5/xxl/)

# How to Cite

If you extend or use this work, please cite the [paper][paper] where it was
introduced:

```
@misc{xue2021byt5,
    title={ByT5: Towards a token-free future with pre-trained byte-to-byte models},
    author={Linting Xue and Aditya Barua and Noah Constant and Rami Al-Rfou and Sharan Narang and Mihir Kale and Adam Roberts and Colin Raffel},
    year={2021},
    eprint={2105.13626},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

[paper]: https://arxiv.org/abs/2105.13626

This is not an officially supported Google product.
