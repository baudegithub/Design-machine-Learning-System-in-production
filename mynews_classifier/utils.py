import os
from typing import Tuple

import pandas as pd
import tensorflow as tf


def set_experiment_dirs(base_dir: str) -> Tuple[str, str, str]:
    """Return data_dir, model_dir, vocab_dir for the project."""
    base_dir = base_dir.rstrip("/")
    return (
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "model"),
        os.path.join(base_dir, "vocab"),
    )


def df_to_tfdata(
    df: pd.DataFrame,
    topic_lookup: tf.keras.layers.StringLookup,
    title_preprocessor: tf.keras.layers.TextVectorization,
    batch_size: int = 32,
    buffer_size: int = 1000,
    shuffle: bool = False,
):
    """Convert a pandas dataframe to a tf.data.Dataset using the notebook logic."""
    if "title" not in df.columns or "topic" not in df.columns:
        raise ValueError("DataFrame must contain 'title' and 'topic' columns.")

    titles = df["title"].astype(str)
    labels = df["topic"].astype(str)

    sequences = title_preprocessor(titles)
    encoded_labels = topic_lookup(labels)

    dataset = tf.data.Dataset.from_tensor_slices((sequences, encoded_labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)

    return dataset


def model_reset_weights(model):
    """Reset weights for a Keras model, mirroring notebook helper logic."""
    for ix, layer in enumerate(model.layers):
        if hasattr(layer, "kernel_initializer") and hasattr(layer, "bias_initializer"):
            weight_initializer = layer.kernel_initializer
            bias_initializer = layer.bias_initializer
            old_weights, old_biases = layer.get_weights()
            layer.set_weights([
                weight_initializer(shape=old_weights.shape),
                bias_initializer(shape=old_biases.shape),
            ])

        if hasattr(layer, "embeddings_initializer"):
            embeddings_initializer = layer.embeddings_initializer
            embedding_weights = layer.get_weights()[0]
            layer.set_weights([
                embeddings_initializer(shape=embedding_weights.shape)
            ])

    return model


def save_data(df: pd.DataFrame, data_dir: str, filename: str):
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(f"{data_dir}/{filename}", index=False)


def save_vocab(tokenizer, vocab_dir: str):
    os.makedirs(vocab_dir, exist_ok=True)
    tokenizer.save_assets(vocab_dir)


def save_labels(topic_lookup, vocab_dir: str):
    os.makedirs(vocab_dir, exist_ok=True)
    labels = topic_lookup.get_vocabulary()
    with open(f"{vocab_dir}/labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(str(w) for w in labels))
