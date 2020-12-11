# Voir >>> https://www.tensorflow.org/tutorials/text/text_generation
# TensorFlow IMPLEMENTATION
import tensorflow as tf

# Import NumPy for shaping datasets and other utils
import numpy as np
import os
import time

# ---- DATA PRE-PROCESSING
# Get File to use as training dataset
path_to_file = 'data/corpus.txt'
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text)) # vocab is the number of unique characters in the dataset

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum sequence length you want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
# # Printing train example
# for i in char_dataset.take(5):
#     print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
# Printing sequences
# for item in sequences.take(5):
#     print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# ---- CREATING TRAINING BATCH
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# print(dataset)
