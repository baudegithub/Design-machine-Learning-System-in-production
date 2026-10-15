"""
Utility helpers used by the experiment runner.
This file provides minimal replacements if lab_utils isn't present in the environment.
"""

import os
import tensorflow as tf
import pandas as pd


def set_experiment_dirs(base_dir: str):
    """Return (data_dir, model_dir, vocab_dir). Mirrors the notebook helper."""
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "model")
    vocab_dir = os.path.join(base_dir, "vocab")
    return data_dir, model_dir, vocab_dir


def df_to_tfdata(df: pd.DataFrame, topic_lookup: tf.keras.layers.StringLookup, title_preprocessor: tf.keras.layers.TextVectorization, batch_size: int = 128):
    """Convert a pandas dataframe with columns ['title', 'topic'] to a batched tf.data.Dataset.

    This matches the preprocessing used in the original notebook: titles are vectorized with
    the provided TextVectorization instance and topics are converted with StringLookup.
    """

    titles = df['title'].astype(str).tolist()
    topics = df['topic'].astype(str).tolist()

    # Convert topics to integer indices using the lookup layer
    topic_indices = topic_lookup(tf.constant(topics)).numpy()

    # Vectorize titles (TextVectorization accepts tf.constant or plain python str)
    title_tensors = title_preprocessor(tf.constant(titles))

    ds = tf.data.Dataset.from_tensor_slices((title_tensors, topic_indices))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
