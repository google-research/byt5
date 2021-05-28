Instructions for how to set sequence length and max decode length.

1.  The input length is generally set to a number that is a power of two and
    larger than 99.9 percentile sequence length in train, dev, and test.

    Exception: The QA tasks of XQuAD, TyDiQA, MLQA uses inputs length of 4096.
    99.9 percentile is much longer than 4096, but we observe that increasing
    beyond 4096 doesn't improve performance.

2.  Target length is affected by packing. We sampled 100 packed examples each
    from train, valid, and test, and take the max packed target length. See
    `compute_target_length.py`.

3.  Set the max decode lengths to cover the max target length in dev/test set,
    so the metrics on tensorboard are accurate.

