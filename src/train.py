"""
Train a title-based news classifier.

Usage:
  python -m src.train --base-dir . --epochs 5 --batch-size 128 --save-dir models/saved_model

This script recreates a simple model similar to the prototype and saves a SavedModel for evaluation and prediction.
"""

import argparse
import os
import uuid
from datetime import datetime

import pandas as pd
import tensorflow as tf

from .utils import set_experiment_dirs, df_to_tfdata


def build_model(vocab_size: int, max_length: int, num_labels: int, embedding_dim: int = 24):
    inputs = tf.keras.Input(shape=(max_length,), dtype="int32", name="title_tokens")
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(inputs)
    x = tf.keras.layers.Dense(24, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def train(base_dir: str, epochs: int, batch_size: int, save_dir: str, max_length: int, vocab_size: int):
    data_dir, model_dir, vocab_dir = set_experiment_dirs(base_dir)

    # Load data
    train_path = os.path.join(data_dir, "train_data.csv") if os.path.exists(os.path.join(data_dir, "train_data.csv")) else os.path.join(base_dir, "train_data.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train_data.csv not found under {data_dir} or {base_dir}")

    df = pd.read_csv(train_path)

    # Create label lookup
    labels_path = os.path.join(vocab_dir, "labels.txt") if os.path.exists(os.path.join(vocab_dir, "labels.txt")) else os.path.join(base_dir, "labels.txt")
    topic_lookup = tf.keras.layers.StringLookup(vocabulary=labels_path, num_oov_indices=0)
    num_labels = len(topic_lookup.get_vocabulary())

    # Text preprocessor
    title_preprocessor = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=max_length)
    # load_assets expects a directory containing 'vocabulary.txt' (or accepts the directory the notebook used)
    assets_dir = vocab_dir if os.path.isdir(vocab_dir) else base_dir
    title_preprocessor.load_assets(assets_dir)

    # Map df titles and topics to tensors
    titles = df['title'].astype(str).tolist()
    topics = df['topic'].astype(str).tolist()
    topic_indices = topic_lookup(tf.constant(topics)).numpy()
    title_tensors = title_preprocessor(tf.constant(titles))

    dataset = tf.data.Dataset.from_tensor_slices((title_tensors, topic_indices))
    dataset = dataset.shuffle(buffer_size=10000, seed=42)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build and compile model
    model = build_model(vocab_size, max_length, num_labels)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    model.summary()

    # Train
    run_id = uuid.uuid4().hex[:8]
    print(f"Starting training run {run_id} — epochs={epochs} batch_size={batch_size}")
    model.fit(dataset, epochs=epochs)

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    print(f"Saved model to {save_dir}")

    return run_id


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='.', help='Experiment directory or repo root')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--save-dir', default='models/saved_model')
    parser.add_argument('--max-length', type=int, default=20)
    parser.add_argument('--vocab-size', type=int, default=10000)
    args = parser.parse_args(argv)

    train(args.base_dir, args.epochs, args.batch_size, args.save_dir, args.max_length, args.vocab_size)


if __name__ == '__main__':
    main()
