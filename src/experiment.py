"""
Utility helpers used by the experiment runner.
This file resolves experiment folders in a flexible way so the project can run
from the original notebook layout (E1/data, E1/model, E1/vocab) or directly from
repo root if the files are stored there.
"""

import os
import tensorflow as tf
import pandas as pd


def set_experiment_dirs(base_dir: str):
    """Return (data_dir, model_dir, vocab_dir).

    Preferred layout:
        <base_dir>/data
        <base_dir>/model
        <base_dir>/vocab

    Fallback layout (matches this repo as checked in):
        <base_dir>/train_data.csv
        <base_dir>/test_data.csv
        <base_dir>/saved_model.pb
        <base_dir>/vocabulary.txt
        <base_dir>/labels.txt
    """
    candidate_data_dir = os.path.join(base_dir, "data")
    candidate_model_dir = os.path.join(base_dir, "model")
    candidate_vocab_dir = os.path.join(base_dir, "vocab")

    if os.path.isdir(candidate_data_dir) and os.path.isdir(candidate_model_dir) and os.path.isdir(candidate_vocab_dir):
        return candidate_data_dir, candidate_model_dir, candidate_vocab_dir

    # Fallback: use repo root for the actual notebook asset layout
    if os.path.exists(os.path.join(base_dir, "train_data.csv")) and os.path.exists(os.path.join(base_dir, "test_data.csv")):
        data_dir = base_dir
    else:
        data_dir = candidate_data_dir

    if os.path.exists(os.path.join(base_dir, "saved_model.pb")) or os.path.exists(os.path.join(base_dir, "saved_model")):
        model_dir = base_dir
    else:
        model_dir = candidate_model_dir

    if os.path.exists(os.path.join(base_dir, "vocabulary.txt")) and os.path.exists(os.path.join(base_dir, "labels.txt")):
        vocab_dir = base_dir
    else:
        vocab_dir = candidate_vocab_dir

    return data_dir, model_dir, vocab_dir


def df_to_tfdata(df: pd.DataFrame, topic_lookup: tf.keras.layers.StringLookup, title_preprocessor: tf.keras.layers.TextVectorization, batch_size: int = 128):
    """Convert a pandas dataframe with columns ['title', 'topic'] to a batched tf.data.Dataset."""
    titles = df['title'].astype(str).tolist()
    topics = df['topic'].astype(str).tolist()

    topic_indices = topic_lookup(tf.constant(topics)).numpy()
    title_tensors = title_preprocessor(tf.constant(titles))

    ds = tf.data.Dataset.from_tensor_slices((title_tensors, topic_indices))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
